import os
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from textblob import TextBlob
import random
import time

app = FastAPI()

class Offer(BaseModel):
    price: float
    message: str

class NegotiationState(BaseModel):
    product: str
    min_price: float
    max_price: float
    current_price: float
    offers: List[Offer]

# Global variable to store negotiation state
negotiation_state = None

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)

def get_ai_response(prompt, client: OpenAI, max_retries=3, delay=1):
    # for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant in a price negotiation."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # if attempt == max_retries - 1:
                # If all retries failed, use a fallback response
                print(e)
                return fallback_response(prompt)
            # time.sleep(delay)  # Wait before retrying

def fallback_response(prompt):
    # Implement a simple rule-based response when OpenAI API is unavailable
    if "offer" in prompt.lower():
        return "I appreciate your offer. However, we need to consider our costs. Could you possibly increase your offer slightly?"
    else:
        return "Thank you for your interest. Let's continue our negotiation. What price did you have in mind?"

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

@app.post("/start_negotiation")
async def start_negotiation(
    product: str = Query(..., description="The product being negotiated"),
    min_price: float = Query(..., description="The minimum acceptable price"),
    max_price: float = Query(..., description="The maximum starting price"),
    client: OpenAI = Depends(get_openai_client)
):
    global negotiation_state
    negotiation_state = NegotiationState(
        product=product,
        min_price=min_price,
        max_price=max_price,
        current_price=max_price,
        offers=[]
    )
    return {"message": f"Negotiation started for {product}. Initial price: ${max_price}"}

@app.post("/make_offer")
async def make_offer(offer: Offer, client: OpenAI = Depends(get_openai_client)):
    global negotiation_state
    if not negotiation_state:
        raise HTTPException(status_code=400, detail="Negotiation not started")

    negotiation_state.offers.append(offer)
    sentiment = analyze_sentiment(offer.message)

    if offer.price >= negotiation_state.current_price:
        return {"message": "Offer accepted!", "final_price": offer.price}

    if offer.price < negotiation_state.min_price:
        return {"message": "Offer too low. Negotiation ended."}

    # Adjust price based on sentiment and offer
    price_range = negotiation_state.max_price - negotiation_state.min_price
    if sentiment > 0:
        price_reduction = random.uniform(0.05, 0.1) * price_range
    else:
        price_reduction = random.uniform(0.01, 0.05) * price_range

    new_price = max(negotiation_state.current_price - price_reduction, negotiation_state.min_price)
    negotiation_state.current_price = new_price

    prompt = f"""
    The customer is negotiating for {negotiation_state.product}. 
    Their offer: ${offer.price}. Their message: "{offer.message}"
    Our current price: ${new_price:.2f}. 
    Respond to the customer, explaining why we can't accept their offer, 
    and propose our new price of ${new_price:.2f}.
    """

    ai_response = get_ai_response(prompt, client)

    return {"message": ai_response, "current_price": new_price}

@app.get("/negotiation_state")
async def get_negotiation_state():
    if not negotiation_state:
        raise HTTPException(status_code=400, detail="Negotiation not started")
    return negotiation_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)