"""
Professional Theme System for ORACULUM
Dark mode optimized with institutional design standards
"""

import streamlit as st


class OracleTheme:
    """Professional color system and styling"""

    # Color Palette
    COLORS = {
        # Primary Colors
        "primary": "#667eea",
        "primary_dark": "#5568d3",
        "primary_light": "#7c8ff5",
        # Backgrounds
        "bg_primary": "#0E1117",
        "bg_secondary": "#1E2130",
        "bg_tertiary": "#262B3D",
        "bg_card": "#1A1F2E",
        "bg_hover": "#2A3142",
        # Borders
        "border_subtle": "#31354A",
        "border_medium": "#3E4458",
        "border_strong": "#4A5066",
        # Text
        "text_primary": "#FFFFFF",
        "text_secondary": "#B8C5D0",
        "text_muted": "#8792A2",
        "text_disabled": "#5E6678",
        # Semantic Colors
        "success": "#00C853",
        "success_bg": "#1B3A2F",
        "warning": "#FFB300",
        "warning_bg": "#3D3420",
        "error": "#FF5252",
        "error_bg": "#3D2020",
        "info": "#2196F3",
        "info_bg": "#1E2940",
        # Gradients
        "gradient_primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "gradient_success": "linear-gradient(135deg, #00c853 0%, #00e676 100%)",
        "gradient_warning": "linear-gradient(135deg, #ffb300 0%, #ffd54f 100%)",
        "gradient_error": "linear-gradient(135deg, #ff5252 0%, #ff1744 100%)",
        # Tier Colors
        "tier_free": "#9E9E9E",
        "tier_basic": "#CD7F32",
        "tier_pro": "#FFD700",
        "tier_enterprise": "#E8E8E8",
    }

    @staticmethod
    def apply_theme():
        """Apply comprehensive theme styling"""
        st.markdown(
            f"""
        <style>
        /* ===== GLOBAL STYLES ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        /* Hide default Streamlit page navigation */
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        :root {{
            --primary: {OracleTheme.COLORS['primary']};
            --bg-primary: {OracleTheme.COLORS['bg_primary']};
            --bg-secondary: {OracleTheme.COLORS['bg_secondary']};
            --bg-card: {OracleTheme.COLORS['bg_card']};
            --text-primary: {OracleTheme.COLORS['text_primary']};
            --text-secondary: {OracleTheme.COLORS['text_secondary']};
            --border-subtle: {OracleTheme.COLORS['border_subtle']};
            --success: {OracleTheme.COLORS['success']};
            --warning: {OracleTheme.COLORS['warning']};
            --error: {OracleTheme.COLORS['error']};
        }}

        /* ===== MAIN CONTAINER ===== */
        .stApp {{
            background: {OracleTheme.COLORS['bg_primary']};
            color: {OracleTheme.COLORS['text_primary']};
        }}

        .main {{
            background: {OracleTheme.COLORS['bg_primary']};
            padding: 2rem 3rem;
        }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {{
            background: {OracleTheme.COLORS['bg_secondary']};
            border-right: 1px solid {OracleTheme.COLORS['border_subtle']};
            box-shadow: 4px 0 12px rgba(0,0,0,0.1);
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 2rem;
        }}

        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3, h4, h5, h6 {{
            color: {OracleTheme.COLORS['text_primary']} !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }}

        h2 {{
            font-size: 2rem;
            margin-bottom: 0.875rem;
        }}

        h3 {{
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
        }}

        p, span, label, .stMarkdown {{
            color: {OracleTheme.COLORS['text_secondary']};
            line-height: 1.6;
        }}

        /* ===== CARDS ===== */
        .card {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .card:hover {{
            border-color: {OracleTheme.COLORS['border_medium']};
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }}

        .card-header {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {OracleTheme.COLORS['text_muted']};
            margin-bottom: 0.5rem;
        }}

        .card-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {OracleTheme.COLORS['text_primary']};
            margin-bottom: 0.75rem;
        }}

        /* ===== METRICS ===== */
        div[data-testid="stMetric"] {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.3s ease;
        }}

        div[data-testid="stMetric"]:hover {{
            border-color: {OracleTheme.COLORS['primary']};
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
        }}

        div[data-testid="stMetric"] label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {OracleTheme.COLORS['text_muted']};
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 800;
            color: {OracleTheme.COLORS['text_primary']};
        }}

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            font-size: 0.875rem;
            font-weight: 600;
        }}

        /* ===== BUTTONS ===== */
        .stButton > button {{
            background: {OracleTheme.COLORS['bg_secondary']};
            color: {OracleTheme.COLORS['text_primary']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 10px;
            padding: 0.625rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
            font-size: 0.925rem;
        }}

        .stButton > button:hover {{
            background: {OracleTheme.COLORS['bg_hover']};
            border-color: {OracleTheme.COLORS['primary']};
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .stButton > button[kind="primary"] {{
            background: {OracleTheme.COLORS['gradient_primary']};
            border: none;
            color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}

        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }}

        .stButton > button[kind="secondary"] {{
            background: transparent;
            border: 2px solid {OracleTheme.COLORS['primary']};
            color: {OracleTheme.COLORS['primary']};
        }}

        /* ===== INPUTS ===== */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {{
            background: {OracleTheme.COLORS['bg_secondary']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 10px;
            color: {OracleTheme.COLORS['text_primary']};
            padding: 0.75rem 1rem;
            font-size: 0.925rem;
        }}

        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {OracleTheme.COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: {OracleTheme.COLORS['bg_secondary']};
            border-radius: 12px;
            padding: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            color: {OracleTheme.COLORS['text_muted']};
            border: none;
            transition: all 0.2s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: {OracleTheme.COLORS['bg_hover']};
            color: {OracleTheme.COLORS['text_primary']};
        }}

        .stTabs [aria-selected="true"] {{
            background: {OracleTheme.COLORS['primary']} !important;
            color: white !important;
        }}

        /* ===== DATAFRAMES ===== */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
        }}

        /* ===== ALERTS ===== */
        .stAlert {{
            border-radius: 12px;
            border-left-width: 4px;
            padding: 1rem 1.25rem;
        }}

        /* ===== EXPANDERS ===== */
        .streamlit-expanderHeader {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 10px;
            padding: 1rem;
            font-weight: 600;
        }}

        .streamlit-expanderHeader:hover {{
            border-color: {OracleTheme.COLORS['primary']};
        }}

        /* ===== PROGRESS BARS ===== */
        .stProgress > div > div {{
            background: {OracleTheme.COLORS['bg_secondary']};
            border-radius: 10px;
        }}

        .stProgress > div > div > div {{
            background: {OracleTheme.COLORS['gradient_primary']};
            border-radius: 10px;
        }}

        /* ===== SPINNERS ===== */
        .stSpinner > div {{
            border-color: {OracleTheme.COLORS['primary']} transparent transparent transparent;
        }}

        /* ===== CUSTOM CLASSES ===== */
        .hero-section {{
            background: {OracleTheme.COLORS['gradient_primary']};
            border-radius: 20px;
            padding: 4rem 2rem;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        }}

        .feature-card {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 16px;
            padding: 2rem;
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .feature-card:hover {{
            border-color: {OracleTheme.COLORS['primary']};
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.2);
        }}

        .stat-card {{
            background: {OracleTheme.COLORS['bg_card']};
            border: 1px solid {OracleTheme.COLORS['border_subtle']};
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: 800;
            background: {OracleTheme.COLORS['gradient_primary']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .stat-label {{
            font-size: 0.875rem;
            color: {OracleTheme.COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.5rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .badge-success {{
            background: {OracleTheme.COLORS['success_bg']};
            color: {OracleTheme.COLORS['success']};
            border: 1px solid {OracleTheme.COLORS['success']};
        }}

        .badge-warning {{
            background: {OracleTheme.COLORS['warning_bg']};
            color: {OracleTheme.COLORS['warning']};
            border: 1px solid {OracleTheme.COLORS['warning']};
        }}

        .badge-error {{
            background: {OracleTheme.COLORS['error_bg']};
            color: {OracleTheme.COLORS['error']};
            border: 1px solid {OracleTheme.COLORS['error']};
        }}

        /* ===== ANIMATIONS ===== */
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .animate-fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {OracleTheme.COLORS['bg_secondary']};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {OracleTheme.COLORS['border_medium']};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {OracleTheme.COLORS['primary']};
        }}

        /* ===== ICON GLOW ANIMATION ===== */
        @keyframes icon-pulse {{
            0% {{
                filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.4));
                transform: scale(1);
            }}
            50% {{
                filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.8));
                transform: scale(1.05);
            }}
            100% {{
                filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.4));
                transform: scale(1);
            }}
        }}

        .glow-icon {{
            display: inline-block;
            filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.1));
            animation: icon-pulse 3s infinite ease-in-out;
        }}

        .stat-card:hover .glow-icon {{
            animation: icon-pulse 1s infinite ease-in-out;
        }}

        .block-container {{
            padding-top: 1rem !important;
                padding-bottom: 0rem !important;
                max-width: 95% !important; /* Optional: gives a more terminal-like width */
            }}

            [data-testid="stHeader"] {{
            display: none;
            }}

            /* 3. Style the page-header to be flush and clean */
            .page-header {{
            margin-top: 0rem !important;
                margin-bottom: 2rem !important;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--border-subtle);
            }}

            .page-header h1 {{
            padding-top: 0 !important;
                margin-top: 0 !important;
            }}

        /* ===== HIDE STREAMLIT BRANDING ===== */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_page_header(title: str, subtitle: str = None, icon: str = None):
        """Render a professional page header"""
        icon_html = f'<span style="font-size: 3rem; margin-right: 1rem;">{icon}</span>' if icon else ""
        subtitle_html = f'<p style="font-size: 1.125rem; color: var(--text-secondary); margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ""

        st.markdown(
            f"""
            <div class="animate-fade-in" style="margin-bottom: 2rem;">
                <div style="display: flex; align-items: center;">
                    {icon_html}
                    <h1 style="margin: 0;">{title}</h1>
                </div>
                {subtitle_html}
            </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def page_align_top():
        st.markdown(
            """
        <style>
            /* 1. Remove the default Streamlit padding at the top of the main container */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
                max-width: 95% !important; /* Optional: gives a more terminal-like width */
            }

            /* 2. Remove the empty space usually reserved for the Streamlit header/anchor */
            [data-testid="stHeader"] {
                display: none;
            }

            /* 3. Style the page-header to be flush and clean */
            .page-header {
                margin-top: 0rem !important;
                margin-bottom: 2rem !important;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--border-subtle);
            }

            .page-header h1 {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
        </style>
    """,
            unsafe_allow_html=True,
        )
