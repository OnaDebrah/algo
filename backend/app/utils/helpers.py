SECTOR_MAP = {
    # --- Technology (Software, Hardware, Semiconductors, Cloud) ---
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "AVGO": "Technology",
    "CRM": "Technology",
    "QCOM": "Technology",
    "ADBE": "Technology",
    "ORCL": "Technology",
    "CSCO": "Technology",
    "IBM": "Technology",
    "NOW": "Technology",
    "SNPS": "Technology",
    "CDNS": "Technology",
    "ANET": "Technology",
    "NET": "Technology",
    "SNOW": "Technology",
    "MU": "Technology",
    "TXN": "Technology",
    "AMAT": "Technology",
    "LRCX": "Technology",
    "KLAC": "Technology",
    "ASML": "Technology",
    "SHOP": "Technology",
    "CRWD": "Technology",
    "PANW": "Technology",
    "FTNT": "Technology",
    "ZS": "Technology",
    "TEAM": "Technology",
    "MDB": "Technology",
    "DDOG": "Technology",
    # --- Financials (Banks, Insurance, Capital Markets, Fintech) ---
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "PYPL": "Financials",
    "AXP": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "SPGI": "Financials",
    "MCO": "Financials",
    "ICE": "Financials",
    "CME": "Financials",
    "CB": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    "TFC": "Financials",
    "BRK.B": "Financials",
    "MET": "Financials",
    "PRU": "Financials",
    "AIG": "Financials",
    "AJG": "Financials",
    "MMC": "Financials",
    "COIN": "Financials",  # Crypto exchange
    "SOFI": "Financials",
    "SQ": "Financials",  # Could also be Tech
    # --- Energy (Oil, Gas, Equipment, Renewables) ---
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "OXY": "Energy",
    "KMI": "Energy",
    "WMB": "Energy",
    "HAL": "Energy",
    "BKR": "Energy",
    "EQT": "Energy",
    "PXD": "Energy",
    "DVN": "Energy",
    "FANG": "Energy",
    "ET": "Energy",
    "EPD": "Energy",
    "NEP": "Energy",  # Renewables
    # --- Healthcare (Pharma, Biotech, Devices, Managed Care) ---
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "LLY": "Healthcare",
    "MRK": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "BMY": "Healthcare",
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "CVS": "Healthcare",
    "CI": "Healthcare",
    "HUM": "Healthcare",
    "MDT": "Healthcare",
    "SYK": "Healthcare",
    "BDX": "Healthcare",
    "BSX": "Healthcare",
    "ISRG": "Healthcare",
    "VRTX": "Healthcare",
    "REGN": "Healthcare",
    "BIIB": "Healthcare",
    "MRNA": "Healthcare",
    "DXCM": "Healthcare",
    "ILMN": "Healthcare",
    "ZTS": "Healthcare",
    "EW": "Healthcare",
    # --- Consumer Staples (Food, Beverage, Household, Personal Products) ---
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "EL": "Consumer Staples",
    "CL": "Consumer Staples",
    "MO": "Consumer Staples",
    "PM": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "KHC": "Consumer Staples",
    "GIS": "Consumer Staples",
    "SYY": "Consumer Staples",
    "KMB": "Consumer Staples",
    "HSY": "Consumer Staples",
    "ADM": "Consumer Staples",
    "BF.B": "Consumer Staples",
    "STZ": "Consumer Staples",
    "KR": "Consumer Staples",
    # --- Consumer Discretionary (Retail, Auto, Hotels, Leisure) ---
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",  # Often categorized here or Tech
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary",
    "HLT": "Consumer Discretionary",
    "YUM": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",  # Sometimes Staples
    "TJX": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    "DLTR": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "NIO": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary",
    "LCID": "Consumer Discretionary",
    "BBY": "Consumer Discretionary",
    "ETSY": "Consumer Discretionary",
    "EXPE": "Consumer Discretionary",
    "CCL": "Consumer Discretionary",
    "RCL": "Consumer Discretionary",
    # --- Communications (Media, Telecom, Entertainment) ---
    "DIS": "Communications",
    "NFLX": "Communications",
    "TMUS": "Communications",
    "VZ": "Communications",
    "T": "Communications",
    "CMCSA": "Communications",
    "CHTR": "Communications",
    "EA": "Communications",
    "ATVI": "Communications",
    "TTWO": "Communications",
    "SPOT": "Communications",
    "LYV": "Communications",
    "OMC": "Communications",
    "IPG": "Communications",
    # --- Industrials (Aerospace, Defense, Machinery, Logistics) ---
    "BA": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "CAT": "Industrials",
    "UPS": "Industrials",
    "FDX": "Industrials",
    "RTX": "Industrials",
    "LMT": "Industrials",
    "NOC": "Industrials",
    "GD": "Industrials",
    "DE": "Industrials",
    "EMR": "Industrials",
    "ITW": "Industrials",
    "WM": "Industrials",
    "RSG": "Industrials",
    "CSX": "Industrials",
    "UNP": "Industrials",
    "NSC": "Industrials",
    "DAL": "Industrials",
    "UAL": "Industrials",
    "LUV": "Industrials",
    "AAL": "Industrials",
    "JCI": "Industrials",
    "ETN": "Industrials",
    "ROK": "Industrials",
    "FTV": "Industrials",
    "TDG": "Industrials",
    # --- Utilities (Electric, Gas, Water, Renewable) ---
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "AEP": "Utilities",
    "D": "Utilities",
    "EXC": "Utilities",
    "SRE": "Utilities",
    "XEL": "Utilities",
    "WEC": "Utilities",
    "ES": "Utilities",
    "PEG": "Utilities",
    "ED": "Utilities",
    "FE": "Utilities",
    "AES": "Utilities",
    "AWK": "Utilities",
    # --- Real Estate (REITs: Office, Retail, Residential, Industrial, Towers) ---
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    "CCI": "Real Estate",
    "DLR": "Real Estate",
    "O": "Real Estate",
    "SPG": "Real Estate",
    "AVB": "Real Estate",
    "EQR": "Real Estate",
    "WELL": "Real Estate",
    "VTR": "Real Estate",
    "SBAC": "Real Estate",
    "WY": "Real Estate",
    "BXP": "Real Estate",
    "VNO": "Real Estate",
    "IRM": "Real Estate",
    "ARE": "Real Estate",
    # --- Materials (Chemicals, Metals, Mining, Construction) ---
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "ECL": "Materials",
    "NEM": "Materials",
    "FCX": "Materials",
    "SCCO": "Materials",
    "VMC": "Materials",
    "MLM": "Materials",
    "STLD": "Materials",
    "NUE": "Materials",
    "CTVA": "Materials",
    "DD": "Materials",
    "DOW": "Materials",
    "LYB": "Materials",
    "PKG": "Materials",
    "WRK": "Materials",
    "CF": "Materials",
    "MOS": "Materials",
    # --- Popular ETFs (for sector tracking or general market) ---
    "SPY": "ETF",
    "IVV": "ETF",
    "VOO": "ETF",
    "QQQ": "ETF",
    "VTI": "ETF",
    "DIA": "ETF",
    "IWM": "ETF",
    "XLK": "ETF",
    "XLF": "ETF",
    "XLV": "ETF",
    "XLE": "ETF",
    "XLY": "ETF",
    "XLP": "ETF",
    "XLI": "ETF",
    "XLC": "ETF",
    "XLU": "ETF",
    "XLRE": "ETF",
    "XLB": "ETF",
    "ARKK": "ETF",
    "VGT": "ETF",
    "VHT": "ETF",
    "VDC": "ETF",
    "VNQ": "ETF",
}

pairs = [
    {
        "asset_1": "AAPL",
        "asset_2": "MSFT",
        "sector": "Technology",
        "correlation": 0.85,
        "cointegration_pvalue": 0.01,
        "description": "Tech Giants - highly correlated",
    },
    {
        "asset_1": "GOOGL",
        "asset_2": "META",
        "sector": "Technology",
        "correlation": 0.78,
        "cointegration_pvalue": 0.02,
        "description": "Ad-driven tech companies",
    },
    {"asset_1": "JPM", "asset_2": "BAC", "sector": "Financials", "correlation": 0.91, "cointegration_pvalue": 0.005, "description": "Major US banks"},
    {
        "asset_1": "KO",
        "asset_2": "PEP",
        "sector": "Consumer Staples",
        "correlation": 0.82,
        "cointegration_pvalue": 0.015,
        "description": "Beverage companies",
    },
    {"asset_1": "XOM", "asset_2": "CVX", "sector": "Energy", "correlation": 0.89, "cointegration_pvalue": 0.008, "description": "Oil & gas majors"},
    {
        "asset_1": "NVDA",
        "asset_2": "AMD",
        "sector": "Technology",
        "correlation": 0.76,
        "cointegration_pvalue": 0.03,
        "description": "GPU/semiconductor manufacturers",
    },
]

SECTOR_MAPPINGS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "ORCL"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "AMGN", "GILD", "BMY", "CVS"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "F", "GM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "WMB", "OXY", "KMI", "MPC", "VLO"],
    "Industrials": ["HON", "UNP", "CAT", "BA", "LMT", "GE", "MMM", "UPS", "FDX", "DE"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB", "GIS"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "ED", "XEL"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "AVB", "EQR"],
    "Communication": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "TMUS", "CHTR", "LYV"],
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication": "XLC",
}

ALTERNATIVE_FEATURES = [
    "sentiment_score",
    "news_volume",
    "analyst_rating",
    "social_momentum",
    "fear_greed_index",
    "sector_sentiment",
    "market_sentiment",
    "insider_trading",
    "short_interest_change",
]

SOURCE_PRIORITY = {
    # FRED is primary for US data
    "gdp_growth": ["fred", "oecd"],
    "cpi_yoy": ["fred", "bls", "oecd"],
    "unemployment_rate": ["bls", "fred", "oecd"],  # BLS is official for US unemployment
    "fed_funds_rate": ["fred"],
    "10y_treasury_yield": ["fred"],
    "2y10y_spread": ["fred"],  # Calculated from FRED
    "m2_money_supply": ["fred"],
    "industrial_production": ["fred"],
    "housing_starts": ["fred"],
    "retail_sales": ["fred"],
    "consumer_confidence": ["fred", "oecd"],
    "manufacturing_pmi": ["oecd", "fred"],
    "services_pmi": ["oecd"],
    "leading_indicator": ["oecd"],
    "nonfarm_payrolls": ["bls"],
    "avg_hourly_earnings": ["bls"],
    "job_openings": ["bls"],
    "oil_prices": ["fred"],
    "dollar_index": ["fred"],
}

# OECD dataset IDs and query paths
DATASET_MAP = {
    # Composite Leading Indicators (CLI)
    "leading_indicator": "MEI_CLI",  # Main Economic Indicators CLI
    # Business Confidence
    "business_confidence": "BSBCI",  # Business Confidence Index
    # Consumer Confidence
    "consumer_confidence": "CSCICP03",  # Consumer Confidence Index [citation:3]
    # PMI (Purchasing Managers Index)
    "manufacturing_pmi": "MEI_PMI",  # Manufacturing PMI
    # GDP
    "gdp_growth": "QNA",  # Quarterly National Accounts
    # Inflation
    "cpi_yoy": "MEI_PRICES",  # Main Economic Indicators Prices
    # Unemployment
    "unemployment_rate": "MEI_LABOUR",  # Main Economic Indicators Labour
    # Interest Rates
    "long_term_rates": "MEI_FIN",  # Financial Indicators
}

# Country codes
COUNTRIES = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "DEU": "Germany",
    "FRA": "France",
    "JPN": "Japan",
    "CAN": "Canada",
    "AUS": "Australia",
    "CHN": "China",
    "IND": "India",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "ZAF": "South Africa",
    "KOR": "Korea",
    "ITA": "Italy",
    "ESP": "Spain",
    "NLD": "Netherlands",
    "CHE": "Switzerland",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "BEL": "Belgium",
    "AUT": "Austria",
    "PRT": "Portugal",
    "GRC": "Greece",
    "IRL": "Ireland",
    "NZL": "New Zealand",
    "ISR": "Israel",
    "TUR": "Turkey",
    "POL": "Poland",
    "CZE": "Czechia",
    "HUN": "Hungary",
    "SVK": "Slovak Republic",
    "SVN": "Slovenia",
    "LUX": "Luxembourg",
    "EST": "Estonia",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "CHL": "Chile",
    "COL": "Colombia",
    "CRI": "Costa Rica",
}

# FRED series IDs for your required indicators [citation:10]
SERIES_MAP = {
    # GDP and Growth
    "gdp_growth": "GDPC1",  # Real GDP
    "gdp_current": "GDP",  # Nominal GDP
    "gdp_potential": "GDPPOT",  # Real Potential GDP
    # Inflation
    "cpi_yoy": "CPIAUCSL",  # CPI All Items [citation:10]
    "cpi_core": "CPILFESL",  # Core CPI
    "ppi": "PPIACO",  # Producer Price Index
    "pce": "PCEPI",  # Personal Consumption Expenditures
    # Labor Market
    "unemployment_rate": "UNRATE",  # Unemployment Rate [citation:10]
    "nonfarm_payrolls": "PAYEMS",  # Nonfarm Payrolls
    "labor_force_participation": "CIVPART",
    "initial_claims": "ICSA",  # Weekly Unemployment Claims
    "job_openings": "JTSJOL",  # JOLTS Job Openings
    # Interest Rates [citation:10]
    "fed_funds_rate": "FEDFUNDS",  # Effective Federal Funds Rate
    "10y_treasury_yield": "GS10",  # 10-Year Treasury Rate
    "2y_treasury_yield": "GS2",  # 2-Year Treasury Rate
    "5y_treasury_yield": "GS5",  # 5-Year Treasury Rate
    "30y_treasury_yield": "GS30",  # 30-Year Treasury Rate
    "corporate_aaa": "AAA",  # Moody's AAA Corporate Bond Yield
    "corporate_baa": "BAA",  # Moody's BAA Corporate Bond Yield
    # Money Supply
    "m2_money_supply": "M2SL",  # M2 Money Stock [citation:10]
    "m1_money_supply": "M1SL",  # M1 Money Stock
    "monetary_base": "AMBSL",  # St. Louis Adjusted Monetary Base
    # Housing
    "housing_starts": "HOUST",  # Housing Starts
    "building_permits": "PERMIT",  # Building Permits
    "new_home_sales": "HSN1F",  # New Home Sales
    "existing_home_sales": "EXHOSLUSM495S",
    # Consumer
    "consumer_confidence": "UMCSENT",  # University of Michigan Consumer Sentiment
    "retail_sales": "RSXFS",  # Retail Sales
    "personal_income": "PI",  # Personal Income
    "personal_savings_rate": "PSAVERT",  # Personal Savings Rate [citation:10]
    # Industrial
    "industrial_production": "INDPRO",  # Industrial Production Index
    "capacity_utilization": "TCU",  # Capacity Utilization
    "manufacturing_pmi": "NAPM",  # ISM Manufacturing PMI
    # International
    "dollar_index": "DTWEXB",  # Trade Weighted Dollar Index
    "oil_prices": "DCOILWTICO",  # Crude Oil Prices
    "trade_balance": "BOPGSTB",  # Trade Balance
}

# BLS series IDs [citation:8]
BLS_SERIES_MAP = {
    # CPI Detailed [citation:8]
    "cpi_all": "CUSR0000SA0",  # CPI-U All Items
    "cpi_core": "CUSR0000SA0L1E",  # CPI-U All Items Less Food and Energy
    "cpi_energy": "CUSR0000SA0E",  # CPI-U Energy
    "cpi_food": "CUSR0000SAF1",  # CPI-U Food
    "cpi_shelter": "CUSR0000SAH1",  # CPI-U Shelter
    # Employment [citation:2]
    "employment_total": "CES0000000001",  # Total Nonfarm Employment
    "employment_private": "CES0500000001",  # Total Private Employment
    "unemployment_rate": "LNS14000000",  # Unemployment Rate [citation:5]
    "labor_force": "LNS11000000",  # Civilian Labor Force [citation:5]
    "employment_level": "LNS12000000",  # Employment Level [citation:5]
    # Wages [citation:8]
    "avg_hourly_earnings": "CES0500000003",  # Average Hourly Earnings
    "avg_weekly_earnings": "CES0500000011",  # Average Weekly Earnings
    "employment_cost_index": "CIS2020000000000I",  # ECI Wages and Salaries
    # Productivity
    "productivity": "PRS85006092",  # Nonfarm Business Productivity
    "unit_labor_costs": "PRS85006112",  # Unit Labor Costs
    # Job Openings [citation:2]
    "job_openings": "JTS000000000000000JOL",
    "quits_rate": "JTS000000000000000QUR",
    "layoffs_rate": "JTS000000000000000JOR",
}

FUNDAMENTAL_FACTORS = [
    "revenue_growth_qq",
    "revenue_growth_yy",
    "eps_growth_qq",
    "eps_growth_yy",
    "roe",
    "roa",
    "roic",
    "pe_ratio",
    "forward_pe",
    "peg_ratio",
    "pb_ratio",
    "ps_ratio",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "operating_margin",
    "net_margin",
    "free_cash_flow_yield",
    "dividend_yield",
    "payout_ratio",
    "rsi_14d",
    "macd_signal",
    "bb_position",
]

MACRO_INDICATORS = [
    "gdp_growth",
    "industrial_production",
    "cpi_mom",
    "cpi_yoy",
    "ppi_mom",
    "unemployment_rate",
    "nonfarm_payrolls",
    "consumer_confidence",
    "manufacturing_pmi",
    "services_pmi",
    "housing_starts",
    "retail_sales_mom",
    "retail_sales_yoy",
    "fed_funds_rate",
    "10y_treasury_yield",
    "2y10y_spread",
    "corporate_spread",
    "vix",
    "m2_money_supply",
]
