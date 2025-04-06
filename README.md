# Groq Voice AI Web and Phone Bot

A conversational agent built with Pipecat, powered by Groq's APIs and Llama 4. Ask it about the weather!

You can deploy this bot to Pipecat Cloud and optionally connect it to Twilio to make it available by phone.

## Configuration

Rename the `env.example` file to `.env` and set the following:

- `GROQ_API_KEY` to use Groq inference (obviously)
- `DAILY_API_KEY` Required to run the bot locally. Get it from your Pipecat Cloud Dashboard: `https://pipecat.daily.co/<your-org-id>/settings/daily`

You'll need a Docker Hub account to deploy. You'll also need a Twilio account if you want to call your bot.

## Set up local environment

Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
## Run the bot
python bot.py
```

To talk to the bot, copy the URL that prints out in the console and open it in your browser. For example:

```
2025-04-05 19:06:13.018 | INFO     | __main__:local:191 - Starting local bot
2025-04-05 19:06:13.243 | INFO     | runner:configure_with_args:78 - Daily room URL: https://cloud-cda71f85b9fe49cdb0f662731f1b3fe7.daily.co/7oQDzyeBVYVYZJCAEBFg
```

## Deploying to Pipecat Cloud

Taken from the [Pipecat Cloud Quickstart](https://docs.pipecat.io/guides/pipecat-cloud/quickstart/).

Build the docker image:

```bash
docker build --platform=linux/arm64 -t groq-llama:latest .
docker tag groq-llama:latest your-username/groq-llama:0.1
docker push your-username/groq-llama:0.1
```

Deploy it to Pipecat Cloud:

You will either need to set your Docker repository to be public, or [provide credentials](https://docs.pipecat.daily.co/agents/deploy#using-pcc-deploy-toml) so Pipecat Cloud can pull from your private repository.

```
pcc auth login # to authenticate
pcc secrets set groq-llama-secrets --file .env # to store your environment variables
pcc deploy groq-llama your-username/groq-llama:0.1 --secrets groq-llama-secrets
```

## Configuring Twilio support

To connect this agent to Twilio:

1. [Purchase a number from Twilio](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console), if you haven't already

2. Collect your Pipecat Cloud organization name:

```bash
pcc organizations list
```

You'll use this information in the next step.

3. Create a [TwiML Bin](https://help.twilio.com/articles/360043489573-Getting-started-with-TwiML-Bins):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://api.pipecat.daily.co/ws/twilio">
      <Parameter name="_pipecatCloudServiceHost" value="AGENT_NAME.ORGANIZATION_NAME"/>
    </Stream>
  </Connect>
</Response>
```

where:

- AGENT_NAME is your agent's name (the name you used when deploying)
- ORGANIZATION_NAME is the value returned in the previous step

In this case, it will look something like `value="groq-llama.level-gorilla-gray-123"`.

4. Assign the TwiML Bin to your phone number:

- Select your number from the Twilio dashboard
- In the `Configure` tab, set `A call comes in` to `TwiML Bin`
- Set `TwiML Bin` to the Bin you created in the previous step
- Save your configuration

Now call your Twilio number, and you should be connected to your bot!

## Customizing the Bot

### Changing the Bot Personality

Modify the system prompt in `bot.py`:

```python
    instructions="""You are a helpful and friendly AI...
```

### Adding more function calls

Search for `get_current_weather` in the codebase to find where the existing function calls are registered. Learn all about Pipecat function calling [here](https://docs.pipecat.io/guides/function-calling/).
