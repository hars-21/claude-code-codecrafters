use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, fs, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let mut messages: Vec<Value> = Vec::new();
    messages.push(json!({
            "role": "user",
            "content": args.prompt
        }
    ));
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
                    if let Value::String(path) = &args["file_path"] {
                        let content = fs::read_to_string(path).unwrap();
                        std::fs::write(path, &content)?;
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": format!("Successfully write {} to {}", content, path)
                        }));
                    } else {
                        panic!("file_path must be a string")
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
