import requests

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    # Register
    r = requests.post(f"{BASE_URL}/auth/register", json={
        "email": "test@test.com", "password": "password", "business_name": "TestCorp"
    })
    print("Register:", r.status_code, r.text)

    # Login
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "test@test.com", "password": "password"
    })
    print("Login:", r.status_code, r.text)
    
    if r.status_code != 200:
        return
        
    token = r.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Check balance
    r = requests.get(f"{BASE_URL}/accounts", headers=headers)
    print("Accounts:", r.status_code, len(r.json()) if r.status_code==200 else r.text)
    
    # Block Card
    r = requests.get(f"{BASE_URL}/cards", headers=headers)
    print("Cards:", r.status_code)
    if r.status_code == 200 and r.json():
        card_id = r.json()[0]['id']
        r = requests.post(f"{BASE_URL}/cards/{card_id}/block", headers=headers)
        print("Block Card API:", r.status_code, r.json())
        
    # Test AI Assistant
    r = requests.post(f"{BASE_URL}/assistant/chat", headers=headers, json={
        "message": "Show my account balance."
    })
    print("AI Assistant (Balance):", r.status_code, r.json())

    r = requests.post(f"{BASE_URL}/assistant/chat", headers=headers, json={
        "message": "Block my card ending 4821.",
        "privacy_sensitive": True
    })
    print("AI Assistant (Privacy Sensitive):", r.status_code, r.json())

if __name__ == "__main__":
    run_tests()
