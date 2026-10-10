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
        "content": format!("You have access to the following skills:\n\n{skill_list}")
    }));

    if let Some(command) = args.prompt.trim().strip_prefix('/') {
        let (skill_name, skill_arguments) = command
            .split_once(char::is_whitespace)
            .map(|(name, arguments)| (name, arguments.trim()))
            .unwrap_or((command, ""));
        let skill = skills
            .iter()
            .find(|skill| skill.name == skill_name)
            .ok_or_else(|| format!("Skill /{skill_name} not found"))?;
        messages.push(json!({
                "role": "user",
                "content": skill.body.replace("$ARGUMENTS", skill_arguments)
            }
        ));
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
            let tool_call = match &tools[0]["function"] {
                Value::Object(tool) => tool,
                _ => panic!("Invalid tool call"),
            };
            let tool_call_id = match &tools[0]["id"] {
                Value::String(id) => id,
                _ => panic!("Tool call id not provided or not string"),
            };
            let tool = match &tool_call["name"] {
                Value::String(name) => name,
                _ => panic!("Tool name must be a string"),
            };
            let args: Value = match &tool_call["arguments"] {
                Value::String(raw) => serde_json::from_str(raw.as_str()).unwrap(),
                _ => panic!("Invalid arguments"),
            };
            match tool.as_str() {
                "Read" => {
                    if let Value::String(path) = &args["file_path"] {
                        let content = fs::read_to_string(path).unwrap();
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content,
                        }))
                    } else {
                        panic!("file_path must be a string")
                    }
                }

                "Write" => {
                    if let (Value::String(path), Value::String(content)) =
                        (&args["file_path"], &args["content"])
                    {
                        std::fs::write(path, content)?;
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": format!("Successfully write {} to {}", content, path)
                        }));
                    } else {
                        panic!("file_path and content must be strings")
                    }
                }

                "Bash" => {
                    if let Value::String(command) = &args["command"] {
                        if command.trim().is_empty() {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": "There is no command to execute!!!"
                            }));
                            // Skip execution for empty command
                            continue;
                        }
                        // Execute the command using a shell to support complex commands and quoting
                        let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
                        let output = Command::new(shell)
                            .arg("-c")
                            .arg(command)
                            .stdin(Stdio::null())
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .output()?;
                        let stdout = String::from_utf8_lossy(&output.stdout);

                        if !stdout.trim().is_empty() {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": stdout
                            }));
                        } else {
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": stderr
                            }));
                        }
                    } else {
                        panic!("command must be a string")
                    }
                }

                _ => panic!("Tool not implemented"),
            };
        } else {
            if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
                println!("{}", content);
                break;
            }
        }
    }

    Ok(())
}
