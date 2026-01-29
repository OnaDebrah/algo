from streamlit.auth.auth_manager import AuthManager, UserTier


def create_admin():
    auth = AuthManager()

    result = auth.register_user(
        username="admin",
        email="admin@yourplatform.com",
        password="Ona@oraculum1",  # CHANGE THIS!
        tier=UserTier.ENTERPRISE,
    )

    if result["success"]:
        print("✅ Admin user created successfully")
        print("Username: admin")
        print("Password: Ona@oraculum1!")
        print("⚠️  CHANGE THE PASSWORD IMMEDIATELY!")
    else:
        print(f"❌ Failed: {result['message']}")


if __name__ == "__main__":
    create_admin()
