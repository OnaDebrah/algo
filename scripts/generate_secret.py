import secrets

if __name__ == "__main__":
    secret_key = secrets.token_hex(32)
    print(f"JWT_SECRET_KEY={secret_key}")
