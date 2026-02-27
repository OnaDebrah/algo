"""
Email templates for crash notifications
"""

from typing import List


class CrashEmailTemplates:
    """HTML email templates for crash alerts"""

    @staticmethod
    def get_crash_template(
        probability: float,
        lead_time_days: int,
        risk_factors: List[str],
        recommended_action: str,
        model_confidence: float,
        intensity: str,
        dashboard_url: str,
    ) -> str:
        """Get HTML email template for crash alert"""

        if probability >= 0.7:
            primary_color = "#dc3545"  # Red
            badge_text = "HIGH RISK"
            badge_color = "#dc3545"
        elif probability >= 0.4:
            primary_color = "#fd7e14"  # Orange
            badge_text = "MODERATE RISK"
            badge_color = "#fd7e14"
        else:
            primary_color = "#ffc107"  # Yellow
            badge_text = "ELEVATED RISK"
            badge_color = "#ffc107"

        risk_factors_html = "".join([f'<li style="margin-bottom: 8px; color: #555;">{factor}</li>' for factor in risk_factors])

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: {primary_color};
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                .header p {{
                    margin: 10px 0 0;
                    font-size: 18px;
                    opacity: 0.95;
                }}
                .content {{
                    background-color: #f8f9fa;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .badge {{
                    display: inline-block;
                    background-color: {badge_color};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin: 25px 0;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 32px;
                    font-weight: 700;
                    color: {primary_color};
                    line-height: 1.2;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .section h2 {{
                    font-size: 20px;
                    font-weight: 600;
                    margin: 0 0 15px;
                    color: #333;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .risk-factors {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                }}
                .risk-factors ul {{
                    margin: 10px 0 0;
                    padding-left: 20px;
                }}
                .action-box {{
                    background: {primary_color}10;
                    border-left: 4px solid {primary_color};
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .action-box p {{
                    margin: 0;
                    font-size: 16px;
                }}
                .button {{
                    display: inline-block;
                    background-color: {primary_color};
                    color: white;
                    text-decoration: none;
                    padding: 15px 30px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                    margin-top: 20px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 14px;
                }}
                .intensity-tag {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                    margin-left: 10px;
                }}
                .severe {{ background: #dc3545; color: white; }}
                .moderate {{ background: #fd7e14; color: white; }}
                .mild {{ background: #28a745; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 Market Crash Alert</h1>
                    <p>Immediate attention recommended</p>
                </div>

                <div class="content">
                    <div style="text-align: center;">
                        <span class="badge">{badge_text}</span>
                    </div>

                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{probability:.1%}</div>
                            <div class="metric-label">Crash Probability</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{lead_time_days}d</div>
                            <div class="metric-label">Expected Timeline</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{model_confidence:.1%}</div>
                            <div class="metric-label">Model Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">
                                {intensity.upper()}
                                <span class="intensity-tag {intensity}">{intensity}</span>
                            </div>
                            <div class="metric-label">Expected Intensity</div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>🔍 Key Risk Factors</h2>
                        <div class="risk-factors">
                            <ul>
                                {risk_factors_html}
                            </ul>
                        </div>
                    </div>

                    <div class="section">
                        <h2>🛡️ Recommended Action</h2>
                        <div class="action-box">
                            <p style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">{recommended_action}</p>
                            <p style="color: #666;">Based on current market conditions and ML model consensus</p>
                        </div>
                    </div>

                    <div style="text-align: center;">
                        <a href="{dashboard_url}" class="button">View Detailed Analysis →</a>
                    </div>

                    <div class="footer">
                        <p>This alert was generated by your Crash Prediction System based on real-time market analysis.</p>
                        <p style="margin-top: 10px;">
                            <a href="{dashboard_url}/settings/notifications" style="color: #666;">Manage notification preferences</a> •
                            <a href="{dashboard_url}/help/crash-alerts" style="color: #666;">Learn about crash alerts</a>
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
