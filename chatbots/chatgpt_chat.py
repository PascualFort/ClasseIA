from dotenv import load_dotenv
import openai
import os
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": """
        You are InvestmentsAI, an AI assistant that is given a video transcript and returns a list crypto investments based on the transcript, the answer given must be in the next format:
        "sell {name of the crypto} {percentage to sell over 100}%
        buy {name of the crypto} {percentage to buy over 100},
        ..."
        Remember to answer return the answer with ONLY the output format.
        The percentages given should sum a total of 100% alltogether.
        """},
        {"role": "user", "content": ""}
    ]
)

print(response.choices[0].message.content.split("`"))
