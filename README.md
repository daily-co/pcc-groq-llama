# Groq Voice AI Web and Phone Starter Kit

## Groq + Llama 4 + Pipecat + function calling + (optionally) Twilio

A conversational agent built with Pipecat, powered by Groq's APIs and Llama 4. Ask it about the weather!

> **Note**: This project temporarily uses a Git submodule checkout of Pipecat at commit `20eebb08e9f059e0800ef3f40429f904becd79d7` instead of the PyPI package. This is a temporary measure until the next Pipecat release. The submodule is located at `./pipecat-subrepo`.
> 
> When cloning this repository or switching to this branch, initialize the submodule with:
> ```bash
> git submodule update --init --recursive
> ```

You can deploy this bot to Pipecat Cloud and optionally connect it to Twilio to make it available by phone.

## Configuration

Rename the `env.example` file to `.env` and set the following:

- `GROQ_API_KEY`

You'll need a Docker Hub account to deploy. You'll also need a Twilio account if you want to call your bot.

## Set up a local development environment

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

To talk to the bot, copy the URL that prints out in the console and open it in your browser. This URL will be `http://localhost:7860` unless you change it in the code for the local development server:

```
 % python bot.py
2025-04-13 21:09:21.484 | INFO     | pipecat:<module>:13 - ᓚᘏᗢ Pipecat 0.0.63 ᓚᘏᗢ
Looking for dist directory at: /Users/khkramer/src/pcc-groq-twilio/venv/lib/python3.11/site-packages/pipecat_ai_small_webrtc_prebuilt/client/dist
2025-04-13 21:09:22.194 | INFO     | __main__:<module>:218 - Starting local development mode
INFO:pipecat-server:Successfully loaded bot from /Users/khkramer/src/pcc-groq-twilio/bot.py, starting web server...
INFO:     Started server process [48084]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:7860 (Press CTRL+C to quit)
```

## Arabic

`byt-en-ar.py` can understand and respond in both English and Arabic. This pipeline splits the response into segments for each language and sends them to separate TTS pipelines.

```bash
python byt-en-ar.py
```

## Client-side UI

To write your own web, iOS, Android, or C++ clients that connect to this bot, see the [Pipecat Client SDK documentation](https://docs.pipecat.ai/client/introduction).

When you test the bot locally, you are talking to the bot using the Pipecat [serverless WebRTC transport](https://docs.pipecat.ai/server/services/transport/small-webrtc). 

You can use this transport in production, but we generally recommend using WebRTC cloud infrastructure like [Daily](https://docs.pipecat.ai/server/services/transport/daily) if you are running voice AI agents in production at scale.

See below for both deploying to Daily's Pipecat Cloud voice agent hosting service and using the Daily WebRTC transport.

## Optionally deploy to Pipecat Cloud

You can host Pipecat agents on any infrastructure that can run Python code and that supports your preferred network transport (WebSockets, WebRTC, etc). See the [Deployment Guide](https://docs.pipecat.ai/guides/deployment/overview) in the Pipecat docs for information and deployment examples.

Pipecat Cloud is a voice agent hosting service built on Daily's [global realtime infrastructure](https://www.daily.co/blog/global-mesh-network/). Pipecat Cloud provides enterprise-grade scalability and management for voice AI agents.

When you use Pipecat Cloud, Daily WebRTC transport for 1:1 audio sessions is [included at no extra cost](https://docs.pipecat.daily.co/pipecat-in-production/daily-webrtc). The code for using Daily WebRTC is in the [bot.py](bot.py) file. So when you deploy this code to Pipecat Cloud your bot will automatically use Daily WebRTC instead of the serverless WebRTC transport.

Here are instructions for deploying to Pipecat Cloud, taken from the [Pipecat Cloud Quickstart](https://docs.pipecat.io/guides/pipecat-cloud/quickstart/).

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

After you've deployed your bot, click on the `groq-llama` agent in the Pipecat Cloud console and then navigate to the `Sandbox` tab to try out your bot.

To learn more about scaling and managing agents using the Pipecat Cloud APIs, see the [Pipecat Cloud documentation](https://docs.pipecat.daily.co/introduction).

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
