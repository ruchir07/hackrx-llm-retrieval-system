import os
import requests
import json
import time

# If you’re using python-dotenv, uncomment these two lines:
# from dotenv import load_dotenv
# load_dotenv()

# Retrieve token from environment
TEAM_TOKEN = os.getenv("HACKRX_TEAM_TOKEN")
if not TEAM_TOKEN:
    raise RuntimeError("HACKRX_TEAM_TOKEN is not set in the environment")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print("Health Check:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_single_question():
    """Test with a single question"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }

    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": ["What is the grace period for premium payment?"]
    }

    print("\n🧪 Testing single question...")
    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print("Answer:", result["answers"][0])
            return True
        else:
            print("❌ Error:", response.text)
            return False

    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_all_questions():
    """Test with all hackathon questions"""
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }

    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }

    print("\n🧪 Testing all 10 hackathon questions...")
    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Total Response Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print("✅ All questions processed successfully!")

            for i, answer in enumerate(result["answers"], 1):
                print(f"\n📝 Q{i}: {data['questions'][i-1]}")
                print(f"💬 A{i}: {answer}")

            return True
        else:
            print("❌ Error:", response.text)
            return False

    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def main():
    print("🚀 Starting API Tests...")

    # Test health endpoint
    if not test_health():
        print("❌ Health check failed. Make sure server is running.")
        return

    # Test single question first
    if test_single_question():
        print("\n✅ Single question test passed!")

        # Ask if user wants to test all questions
        user_input = input("\n🤔 Do you want to test all 10 questions? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            test_all_questions()
    else:
        print("❌ Single question test failed. Check your setup.")

if __name__ == "__main__":
    main()
