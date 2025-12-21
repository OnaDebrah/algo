"""
Setup script to create the project directory structure
Run this script to automatically create all necessary folders and files
"""

import os


def create_directory_structure():
    """Create the complete directory structure for the trading platform"""

    # Define directory structure
    directories = [
        "core",
        "strategies",
        "analytics",
        "alerts",
        "ui",
        "tests",
        "tests/core",
        "tests/strategies",
        "tests/analytics",
        "data",
        "logs",
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

        # Create __init__.py in Python package directories
        if (
            directory != "tests"
            and directory != "data"
            and directory != "logs"
            and not directory.startswith("tests/")
        ):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write(f'"""{directory.capitalize()} package"""\n')
                print(f"  ✓ Created {init_file}")

    print("\n✅ Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Copy the code files into their respective directories")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the application: streamlit run main.py")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# Environmen
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Streamli
.streamlit/

# Data
data/*.csv
data/*.json
!data/.gitkeep

# Tests
.pytest_cache/
.coverage
htmlcov/
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())

    print("\n✓ Created .gitignore")


def create_placeholder_files():
    """Create placeholder files in data and logs directories"""

    # Data directory
    with open("data/.gitkeep", "w") as f:
        f.write("")

    # Logs directory
    with open("logs/.gitkeep", "w") as f:
        f.write("")

    print("✓ Created placeholder files")


def print_file_mapping():
    """Print where each code file should go"""
    print("\n" + "=" * 70)
    print("FILE PLACEMENT GUIDE")
    print("=" * 70)

    file_mapping = {
        "Root Directory": [
            "main.py",
            "config.py",
            "requirements.txt",
            "README.md",
            ".gitignore",
        ],
        "core/": [
            "__init__.py",
            "database.py",
            "data_fetcher.py",
            "risk_manager.py",
            "trading_engine.py",
        ],
        "strategies/": [
            "__init__.py",
            "base_strategy.py",
            "sma_crossover.py",
            "rsi_strategy.py",
            "macd_strategy.py",
            "ml_strategy.py",
        ],
        "analytics/": ["__init__.py", "performance.py"],
        "alerts/": ["__init__.py", "alert_manager.py"],
        "ui/": [
            "__init__.py",
            "dashboard.py",
            "backtest.py",
            "ml_builder.py",
            "configuration.py",
        ],
    }

    for directory, files in file_mapping.items():
        print(f"\n{directory}")
        for file in files:
            print(f"  - {file}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("🚀 Trading Platform Setup\n")

    create_directory_structure()
    create_gitignore()
    create_placeholder_files()
    print_file_mapping()

    print("\n✨ Setup complete! Follow the file placement guide above.")
