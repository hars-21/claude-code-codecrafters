use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process};

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

    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "model": "anthropic/claude-haiku-4.5",
            "messages": [{"role": "user", "content": args.prompt}],
            "tools": [{
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
            }]
        }))
        .await?;

    eprintln!("Logs from your program will appear here!");

    let message = &response["choices"][0]["message"];

    if let Some(tool_calls) = message["tool_calls"].as_array() {
        let tool_call = &tool_calls[0];
        let name = tool_call["function"]["name"].as_str().unwrap();
        let arguments: Value =
            serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap())?;

        if name == "Read" {
            let file_path = arguments["file_path"].as_str().unwrap();
            let contents = std::fs::read_to_string(file_path)?;
            print!("{}", contents);
        }
    } else if let Some(content) = message["content"].as_str() {
        println!("{}", content);
    }

    Ok(())
}
