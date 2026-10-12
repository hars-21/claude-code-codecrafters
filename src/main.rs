use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Stdio};
use std::{env, fs, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[derive(Deserialize)]
struct Skill {
    name: String,
    description: String,
    context: Option<String>,
    #[serde(skip)]
    body: String,
}

fn read_skill(content: &str) -> Result<Option<Skill>, serde_yaml::Error> {
    let mut lines = content.lines();
    let Some(first_line) = lines.next() else {
        return Ok(None);
    };

    if first_line.trim_start_matches('\u{feff}') != "---" {
        return Ok(None);
    }

    let mut frontmatter = Vec::new();
    let mut found_closing_delimiter = false;
    for line in &mut lines {
        if line == "---" {
            found_closing_delimiter = true;
            break;
        }
        frontmatter.push(line);
    }

    if !found_closing_delimiter || frontmatter.is_empty() {
        return Ok(None);
    }

    let mut skill: Skill = serde_yaml::from_str(&frontmatter.join("\n"))?;
    skill.body = lines.collect::<Vec<_>>().join("\n");

    Ok(Some(skill))
}

fn load_skills() -> Result<Vec<Skill>, Box<dyn std::error::Error>> {
    let skills_path = Path::new(".claude/skills");
    let mut skills = Vec::new();

    if !skills_path.is_dir() {
        return Ok(skills);
    }

    for entry in fs::read_dir(skills_path)? {
        let filename = entry?.path().join("SKILL.md");
        if !filename.is_file() {
            continue;
        }

        let contents = fs::read_to_string(&filename)?;
        let skill = read_skill(&contents)?
            .ok_or_else(|| format!("Frontmatter missing from {}", filename.display()))?;

        skills.push(skill);
    }

    Ok(skills)
}

fn render_skill_body(body: &str, arguments: &str) -> String {
    let positional_arguments = arguments.split_whitespace().collect::<Vec<_>>();
    let full_arguments = positional_arguments.join(" ");
    let mut rendered = body.to_string();

    for (index, argument) in positional_arguments.iter().enumerate().rev() {
        rendered = rendered.replace(&format!("$ARGUMENTS[{index}]"), argument);
        rendered = rendered.replace(&format!("${index}"), argument);
    }

    rendered.replace("$ARGUMENTS", &full_arguments)
}

async fn run_subagent(
    client: &Client<OpenAIConfig>,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let response: Value = client
        .chat()
        .create_byot(json!({
            "model": "anthropic/claude-haiku-4.5",
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }))
        .await?;

    response["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "Subagent did not return a response".into())
}

fn render_skills<'a>(
    prompt: &str,
    skills: &'a [Skill],
) -> Result<Vec<(&'a Skill, String)>, Box<dyn std::error::Error>> {
    let tokens = prompt.split_whitespace().collect::<Vec<_>>();
    let mut expanded_skills = Vec::new();
    let mut argument_start = 0;

    for (index, token) in tokens.iter().enumerate() {
        let Some(skill_name) = token.strip_prefix('/') else {
            break;
        };

        let Some(skill) = skills.iter().find(|skill| skill.name == skill_name) else {
            break;
        };

        expanded_skills.push(skill);
        argument_start = index + 1;
    }

    if expanded_skills.is_empty() {
        return Err(format!("Skill {} not found", tokens.first().unwrap_or(&"/")).into());
    }

    let arguments = tokens[argument_start..].join(" ");

    Ok(expanded_skills
        .into_iter()
        .map(|skill| (skill, render_skill_body(&skill.body, &arguments)))
        .collect())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let skills = load_skills()?;

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let skill_list = skills
        .iter()
        .map(|skill| format!("- {}: {}\n", skill.name, skill.description))
        .collect::<String>();

    let client = Client::with_config(config);

    let mut messages: Vec<Value> = Vec::new();

    messages.push(json!({
        "role": "system",
        "content": format!(
            "You have access to the following skills. Use the Skill tool when a skill's description applies to the user's request.\n\n{skill_list}"
        )
    }));

    if args.prompt.trim().starts_with('/') {
        for (skill, skill_body) in render_skills(&args.prompt, &skills)? {
            let content = if skill.context.as_deref() == Some("fork") {
                run_subagent(&client, &skill_body).await?
            } else {
                skill_body
            };
            messages.push(json!({
                "role": "user",
                "content": content
            }));
        }
    } else {
        messages.push(json!({
                "role": "user",
                "content": args.prompt
            }
        ));
    }

    for _ in 0..10 {
        #[allow(unused_variables)]
        let response: Value = client
            .chat()
            .create_byot(json!({
                    "model": "anthropic/claude-haiku-4.5",
                    "messages": messages,
                    "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Skill",
                            "description": "Load the full instructions for a skill when its description applies to the user's request.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "skill_name": {
                                        "type": "string",
                                        "description": "The name of the skill to invoke"
                                    },
                                    "arguments": {
                                        "type": "string",
                                        "description": "Optional arguments to pass to the skill"
                                    }
                                },
                                "required": ["skill_name"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                                "type": "object",
                                "required": ["file_path", "content"],
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path of the file to write to"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content to write to the file"
                                    }
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Execute a shell command",
                            "parameters": {
                                "type": "object",
                                "required": ["command"],
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "The command to execute"
                                    }
                                }
                            }
                        }
                    }
                ]
            }))
            .await?;

        if let Some(tools) = response["choices"][0]["message"]["tool_calls"].as_array()
            && !tools.is_empty()
        {
            messages.push(response["choices"][0]["message"].clone());
            let tool_call_id = tools[0]["id"]
                .as_str()
                .ok_or("Tool call id must be a string")?;
            let tool = tools[0]["function"]["name"]
                .as_str()
                .ok_or("Tool name must be a string")?;
            let args = match tools[0]["function"]["arguments"].as_str() {
                Some(raw) => match serde_json::from_str::<Value>(raw) {
                    Ok(args) => args,
                    Err(error) => {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": format!("Invalid tool arguments: {error}"),
                        }));
                        continue;
                    }
                },
                None => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Invalid tool arguments: expected a JSON string",
                    }));
                    continue;
                }
            };

            let content = match tool {
                "Skill" => match args["skill_name"].as_str() {
                    Some(skill_name) => {
                        match skills.iter().find(|skill| skill.name == skill_name) {
                            Some(skill) => {
                                let body = render_skill_body(
                                    &skill.body,
                                    args["arguments"].as_str().unwrap_or(""),
                                );
                                if skill.context.as_deref() == Some("fork") {
                                    run_subagent(&client, &body).await?
                                } else {
                                    body
                                }
                            }
                            None => format!("Skill /{skill_name} not found"),
                        }
                    }
                    None => "Invalid Skill arguments: skill_name must be a string".to_string(),
                },
                "Read" => match args["file_path"].as_str() {
                    Some(path) => fs::read_to_string(path)
                        .unwrap_or_else(|error| format!("Error reading {path}: {error}")),
                    None => "Invalid Read arguments: file_path must be a string".to_string(),
                },
                "Write" => match (args["file_path"].as_str(), args["content"].as_str()) {
                    (Some(path), Some(file_content)) => match fs::write(path, file_content) {
                        Ok(()) => format!("Successfully wrote {path}"),
                        Err(error) => format!("Error writing {path}: {error}"),
                    },
                    _ => {
                        "Invalid Write arguments: file_path and content must be strings".to_string()
                    }
                },
                "Bash" => match args["command"].as_str() {
                    Some(command) if command.trim().is_empty() => {
                        "There is no command to execute".to_string()
                    }
                    Some(command) => {
                        let shell = env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
                        match Command::new(shell)
                            .arg("-c")
                            .arg(command)
                            .stdin(Stdio::null())
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .output()
                        {
                            Ok(output) if !output.stdout.is_empty() => {
                                String::from_utf8_lossy(&output.stdout).into_owned()
                            }
                            Ok(output) => String::from_utf8_lossy(&output.stderr).into_owned(),
                            Err(error) => format!("Error running command: {error}"),
                        }
                    }
                    None => "Invalid Bash arguments: command must be a string".to_string(),
                },
                _ => format!("Unknown tool: {tool}"),
            };

            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }));
        } else {
            if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
                println!("{}", content);
                break;
            }
        }
    }

    Ok(())
}
