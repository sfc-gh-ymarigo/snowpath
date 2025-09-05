CREATE OR REPLACE FUNCTION generate_fsi_user_journey()
RETURNS TABLE (
    -- Core identifiers
    event_id STRING,
    visitor_id STRING,
    customer_id STRING,
    
    -- Event details
    event_timestamp TIMESTAMP,
    event_type STRING,
    event_category STRING,
    event_action STRING,
    event_label STRING,
    
    -- Page/Screen information (Adobe Analytics style)
    page_name STRING,
    page_url STRING,
    page_type STRING,
    site_section STRING,
    referrer_url STRING,
    
    -- Technical details
    browser STRING,
    browser_version STRING,
    operating_system STRING,
    device_type STRING,
    screen_resolution STRING,
    user_agent STRING,
    ip_address STRING,
    
    -- Geographic data
    country STRING,
    state STRING,
    city STRING,
    zip_code STRING,
    
    -- Page interaction details
    time_on_page INT,
    scroll_depth INT,
    clicks_on_page INT,
    
    -- Banking specific fields
    account_type STRING,
    product_category STRING,
    transaction_amount DECIMAL(12,2),
    channel STRING,
    authentication_method STRING,
    customer_segment STRING,
    
    -- Campaign/Marketing (Adobe Analytics style)
    campaign_id STRING,
    traffic_source STRING,
    medium STRING,
    referrer_domain STRING,
    
    -- Custom dimensions with explicit names
    customer_tenure STRING,
    account_balance_tier STRING,
    product_interest STRING,
    mobile_app_version STRING,
    customer_lifetime_value_tier STRING,
    
    -- Custom events with explicit names
    form_starts INT,
    form_completions INT,
    errors_encountered INT,
    support_interactions INT,
    product_views INT,
    
    -- Additional context
    is_mobile_app BOOLEAN,
    page_load_time_ms INT,
    conversion_flag BOOLEAN,
    revenue_impact DECIMAL(12,2)
)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'generateJourney'
PACKAGES = ('faker')
AS $$
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

class generateJourney:
    def process(self):
        # Define shared event pools that can be used across multiple journeys
        shared_events = {
            'entry_points': [
                'homepage_visit', 'direct_login', 'email_campaign_click', 'search_result_click',
                'social_media_click', 'mobile_app_open', 'branch_referral_visit'
            ],
            'authentication': [
                'login_attempt', 'login_success', 'password_reset_request', 'two_factor_challenge',
                'biometric_auth', 'security_question_prompt', 'account_locked_warning'
            ],
            'account_management': [
                'account_summary_view', 'balance_check', 'transaction_history_view',
                'statement_download', 'account_settings_view', 'profile_update_start',
                'contact_info_update', 'notification_preferences'
            ],
            'product_research': [
                'product_page_visit', 'product_comparison_tool', 'rate_lookup',
                'calculator_use', 'feature_comparison', 'eligibility_checker',
                'testimonial_view', 'faq_browse', 'terms_conditions_view'
            ],
            'application_process': [
                'application_landing', 'application_start', 'personal_info_entry',
                'financial_info_entry', 'document_upload', 'identity_verification',
                'application_review', 'application_submit', 'application_confirmation'
            ],
            'transactional': [
                'transfer_initiate', 'transfer_setup', 'payment_scheduling',
                'payee_management', 'payment_confirmation', 'transaction_receipt_view',
                'recurring_payment_setup', 'payment_history_view'
            ],
            'investment_activities': [
                'portfolio_overview', 'market_dashboard', 'stock_research',
                'trade_preparation', 'order_entry', 'trade_execution',
                'performance_review', 'rebalancing_tools'
            ],
            'support_touchpoints': [
                'help_center_visit', 'search_help_articles', 'faq_section_browse',
                'contact_options_view', 'live_chat_initiate', 'phone_callback_request',
                'support_ticket_creation', 'branch_appointment_booking',
                'agent_interaction', 'issue_escalation', 'resolution_confirmation',
                'feedback_survey'
            ],
            'mobile_specific': [
                'mobile_dashboard', 'quick_balance_check', 'mobile_deposit_camera',
                'location_services_enable', 'push_notification_interaction',
                'app_settings_access', 'biometric_setup'
            ],
            'security_actions': [
                'security_center_visit', 'fraud_alert_review', 'card_management',
                'travel_notification_setup', 'security_settings_update',
                'device_management', 'suspicious_activity_review'
            ],
            'cross_selling': [
                'product_recommendation_view', 'promotional_banner_click',
                'upgrade_offer_consideration', 'additional_product_research',
                'cross_sell_application_start'
            ],
            'exits': [
                'logout', 'session_timeout', 'navigation_away', 'app_background',
                'browser_close', 'phone_call_transfer'
            ]
        }
        
        # Define journey templates with shared events and branching logic
        journey_templates = {
            'new_customer_exploration': {
                'primary_goal': 'account_opening',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_research', random.randint(2, 4)),
                    ('support_touchpoints', random.randint(0, 2)),  # May need help
                    ('application_process', random.randint(3, 7)),
                    ('account_management', random.randint(1, 2)),  # Check new account
                    ('exits', 1)
                ],
                'conversion_rate': 0.6,
                'revenue_range': (100, 500)
            },
            'existing_customer_expansion': {
                'primary_goal': 'product_addition',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(1, 2)),
                    ('account_management', random.randint(2, 3)),
                    ('cross_selling', random.randint(1, 2)),
                    ('product_research', random.randint(1, 3)),
                    ('application_process', random.randint(2, 5)),
                    ('transactional', random.randint(0, 2)),  # May do other banking
                    ('exits', 1)
                ],
                'conversion_rate': 0.45,
                'revenue_range': (200, 2000)
            },
            'loan_shopping_journey': {
                'primary_goal': 'loan_application',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(0, 2)),  # May browse without login
                    ('product_research', random.randint(3, 5)),
                    ('support_touchpoints', random.randint(1, 3)),  # Likely need guidance
                    ('account_management', random.randint(0, 2)),  # Check existing accounts
                    ('application_process', random.randint(2, 6)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.35,
                'revenue_range': (1000, 10000)
            },
            'routine_banking_session': {
                'primary_goal': 'transaction_completion',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(1, 2)),
                    ('account_management', random.randint(2, 4)),
                    ('transactional', random.randint(2, 4)),
                    ('cross_selling', random.randint(0, 1)),  # May see offers
                    ('product_research', random.randint(0, 2)),  # May browse
                    ('exits', 1)
                ],
                'conversion_rate': 0.85,
                'revenue_range': (0, 0)  # No direct revenue
            },
            'investment_management': {
                'primary_goal': 'trade_execution',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(1, 2)),
                    ('investment_activities', random.randint(3, 6)),
                    ('account_management', random.randint(0, 2)),
                    ('support_touchpoints', random.randint(0, 1)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.55,
                'revenue_range': (5, 50)
            },
            'support_resolution': {
                'primary_goal': 'issue_resolution',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(0, 2)),
                    ('support_touchpoints', random.randint(3, 6)),
                    ('account_management', random.randint(1, 3)),  # Review account details
                    ('security_actions', random.randint(0, 2)),  # May involve security
                    ('transactional', random.randint(0, 1)),  # May do transactions
                    ('exits', 1)
                ],
                'conversion_rate': 0.75,
                'revenue_range': (0, 0)
            },
            'mobile_banking_session': {
                'primary_goal': 'mobile_transaction',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('mobile_specific', random.randint(2, 4)),
                    ('account_management', random.randint(1, 3)),
                    ('transactional', random.randint(1, 3)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.80,
                'revenue_range': (0, 0)
            },
            'account_closure_journey': {
                'primary_goal': 'account_closure',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(1, 2)),
                    ('account_management', random.randint(2, 3)),  # Review accounts
                    ('support_touchpoints', random.randint(2, 4)),  # Need help closing
                    ('transactional', random.randint(1, 3)),  # Transfer remaining funds
                    ('security_actions', random.randint(0, 2)),  # Security verification
                    ('support_touchpoints', random.randint(1, 2)),  # Final confirmation
                    ('exits', 1)
                ],
                'conversion_rate': 0.40,  # May retain customer
                'revenue_range': (-500, 0)  # Negative revenue impact
            },
            'security_incident_response': {
                'primary_goal': 'security_resolution',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(1, 3)),  # Multiple auth attempts
                    ('security_actions', random.randint(3, 5)),
                    ('support_touchpoints', random.randint(2, 4)),
                    ('account_management', random.randint(1, 2)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.70,
                'revenue_range': (0, 0)
            },
            'research_abandonment': {
                'primary_goal': 'research_only',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_research', random.randint(2, 5)),
                    ('support_touchpoints', random.randint(0, 2)),
                    ('authentication', random.randint(0, 1)),  # May not even log in
                    ('application_process', random.randint(0, 2)),  # Partial application
                    ('exits', 1)
                ],
                'conversion_rate': 0.05,  # Very low conversion
                'revenue_range': (0, 0)
            }
        }
        
        # Detailed event mappings
        event_details = {
            # Entry points
            'homepage_visit': {'name': 'Homepage', 'url': '/', 'type': 'marketing', 'category': 'navigation'},
            'direct_login': {'name': 'Direct Login', 'url': '/login', 'type': 'authentication', 'category': 'authentication'},
            'email_campaign_click': {'name': 'Email Campaign', 'url': '/campaign-landing', 'type': 'marketing', 'category': 'campaign'},
            'search_result_click': {'name': 'Search Results', 'url': '/search-landing', 'type': 'marketing', 'category': 'acquisition'},
            'social_media_click': {'name': 'Social Media', 'url': '/social-landing', 'type': 'marketing', 'category': 'social'},
            'mobile_app_open': {'name': 'Mobile App Home', 'url': '/app', 'type': 'mobile', 'category': 'mobile_banking'},
            'branch_referral_visit': {'name': 'Branch Referral', 'url': '/branch-referral', 'type': 'marketing', 'category': 'branch'},
            
            # Authentication
            'login_attempt': {'name': 'Login Attempt', 'url': '/login', 'type': 'authentication', 'category': 'authentication'},
            'login_success': {'name': 'Login Success', 'url': '/dashboard', 'type': 'authentication', 'category': 'authentication'},
            'password_reset_request': {'name': 'Password Reset', 'url': '/password-reset', 'type': 'authentication', 'category': 'security'},
            'two_factor_challenge': {'name': '2FA Challenge', 'url': '/2fa-verify', 'type': 'authentication', 'category': 'security'},
            'biometric_auth': {'name': 'Biometric Auth', 'url': '/biometric-login', 'type': 'authentication', 'category': 'security'},
            'security_question_prompt': {'name': 'Security Questions', 'url': '/security-questions', 'type': 'authentication', 'category': 'security'},
            'account_locked_warning': {'name': 'Account Locked', 'url': '/account-locked', 'type': 'security', 'category': 'security'},
            
            # Account Management
            'account_summary_view': {'name': 'Account Summary', 'url': '/accounts', 'type': 'account_management', 'category': 'account_management'},
            'balance_check': {'name': 'Balance Check', 'url': '/accounts/balance', 'type': 'account_management', 'category': 'account_management'},
            'transaction_history_view': {'name': 'Transaction History', 'url': '/accounts/transactions', 'type': 'account_management', 'category': 'account_management'},
            'statement_download': {'name': 'Download Statement', 'url': '/accounts/statements', 'type': 'account_management', 'category': 'account_management'},
            'account_settings_view': {'name': 'Account Settings', 'url': '/settings/account', 'type': 'account_management', 'category': 'account_management'},
            'profile_update_start': {'name': 'Update Profile', 'url': '/profile/edit', 'type': 'account_management', 'category': 'account_management'},
            'contact_info_update': {'name': 'Update Contact Info', 'url': '/profile/contact', 'type': 'account_management', 'category': 'account_management'},
            'notification_preferences': {'name': 'Notification Settings', 'url': '/settings/notifications', 'type': 'account_management', 'category': 'account_management'},
            
            # Product Research
            'product_page_visit': {'name': 'Products Overview', 'url': '/products', 'type': 'marketing', 'category': 'product_research'},
            'product_comparison_tool': {'name': 'Product Comparison', 'url': '/products/compare', 'type': 'tools', 'category': 'product_research'},
            'rate_lookup': {'name': 'Interest Rates', 'url': '/rates', 'type': 'marketing', 'category': 'product_research'},
            'calculator_use': {'name': 'Financial Calculator', 'url': '/tools/calculator', 'type': 'tools', 'category': 'product_research'},
            'feature_comparison': {'name': 'Feature Comparison', 'url': '/products/features', 'type': 'marketing', 'category': 'product_research'},
            'eligibility_checker': {'name': 'Eligibility Check', 'url': '/tools/eligibility', 'type': 'tools', 'category': 'product_research'},
            'testimonial_view': {'name': 'Customer Testimonials', 'url': '/testimonials', 'type': 'marketing', 'category': 'product_research'},
            'faq_browse': {'name': 'FAQ Browse', 'url': '/faq', 'type': 'support', 'category': 'product_research'},
            'terms_conditions_view': {'name': 'Terms & Conditions', 'url': '/legal/terms', 'type': 'legal', 'category': 'product_research'},
            
            # Application Process
            'application_landing': {'name': 'Application Landing', 'url': '/apply', 'type': 'application', 'category': 'application_process'},
            'application_start': {'name': 'Start Application', 'url': '/apply/start', 'type': 'application', 'category': 'application_process'},
            'personal_info_entry': {'name': 'Personal Information', 'url': '/apply/personal', 'type': 'application', 'category': 'application_process'},
            'financial_info_entry': {'name': 'Financial Information', 'url': '/apply/financial', 'type': 'application', 'category': 'application_process'},
            'document_upload': {'name': 'Document Upload', 'url': '/apply/documents', 'type': 'application', 'category': 'application_process'},
            'identity_verification': {'name': 'Identity Verification', 'url': '/apply/verify', 'type': 'application', 'category': 'application_process'},
            'application_review': {'name': 'Review Application', 'url': '/apply/review', 'type': 'application', 'category': 'application_process'},
            'application_submit': {'name': 'Submit Application', 'url': '/apply/submit', 'type': 'application', 'category': 'application_process'},
            'application_confirmation': {'name': 'Application Confirmation', 'url': '/apply/confirmation', 'type': 'application', 'category': 'application_process'},
            
            # Transactional
            'transfer_initiate': {'name': 'Initiate Transfer', 'url': '/transfer', 'type': 'transaction', 'category': 'transactional'},
            'transfer_setup': {'name': 'Transfer Setup', 'url': '/transfer/setup', 'type': 'transaction', 'category': 'transactional'},
            'payment_scheduling': {'name': 'Schedule Payment', 'url': '/payments/schedule', 'type': 'transaction', 'category': 'transactional'},
            'payee_management': {'name': 'Manage Payees', 'url': '/payments/payees', 'type': 'transaction', 'category': 'transactional'},
            'payment_confirmation': {'name': 'Payment Confirmation', 'url': '/payments/confirm', 'type': 'transaction', 'category': 'transactional'},
            'transaction_receipt_view': {'name': 'Transaction Receipt', 'url': '/receipts', 'type': 'transaction', 'category': 'transactional'},
            'recurring_payment_setup': {'name': 'Recurring Payments', 'url': '/payments/recurring', 'type': 'transaction', 'category': 'transactional'},
            'payment_history_view': {'name': 'Payment History', 'url': '/payments/history', 'type': 'transaction', 'category': 'transactional'},
            
            # Investment Activities
            'portfolio_overview': {'name': 'Portfolio Overview', 'url': '/investments', 'type': 'investment', 'category': 'investment_activities'},
            'market_dashboard': {'name': 'Market Dashboard', 'url': '/investments/market', 'type': 'investment', 'category': 'investment_activities'},
            'stock_research': {'name': 'Stock Research', 'url': '/investments/research', 'type': 'investment', 'category': 'investment_activities'},
            'trade_preparation': {'name': 'Trade Preparation', 'url': '/investments/trade-prep', 'type': 'investment', 'category': 'investment_activities'},
            'order_entry': {'name': 'Order Entry', 'url': '/investments/order', 'type': 'investment', 'category': 'investment_activities'},
            'trade_execution': {'name': 'Trade Execution', 'url': '/investments/execute', 'type': 'investment', 'category': 'investment_activities'},
            'performance_review': {'name': 'Performance Review', 'url': '/investments/performance', 'type': 'investment', 'category': 'investment_activities'},
            'rebalancing_tools': {'name': 'Portfolio Rebalancing', 'url': '/investments/rebalance', 'type': 'investment', 'category': 'investment_activities'},
            
            # Support Touchpoints
            'help_center_visit': {'name': 'Help Center', 'url': '/help', 'type': 'support', 'category': 'support_touchpoints'},
            'search_help_articles': {'name': 'Search Help', 'url': '/help/search', 'type': 'support', 'category': 'support_touchpoints'},
            'faq_section_browse': {'name': 'FAQ Section', 'url': '/help/faq', 'type': 'support', 'category': 'support_touchpoints'},
            'contact_options_view': {'name': 'Contact Options', 'url': '/contact', 'type': 'support', 'category': 'support_touchpoints'},
            'live_chat_initiate': {'name': 'Start Live Chat', 'url': '/support/chat', 'type': 'support', 'category': 'support_touchpoints'},
            'phone_callback_request': {'name': 'Request Callback', 'url': '/support/callback', 'type': 'support', 'category': 'support_touchpoints'},
            'support_ticket_creation': {'name': 'Create Support Ticket', 'url': '/support/ticket', 'type': 'support', 'category': 'support_touchpoints'},
            'branch_appointment_booking': {'name': 'Book Branch Appointment', 'url': '/branch/appointment', 'type': 'support', 'category': 'support_touchpoints'},
            'agent_interaction': {'name': 'Agent Interaction', 'url': '/support/agent', 'type': 'support', 'category': 'support_touchpoints'},
            'issue_escalation': {'name': 'Issue Escalation', 'url': '/support/escalate', 'type': 'support', 'category': 'support_touchpoints'},
            'resolution_confirmation': {'name': 'Issue Resolved', 'url': '/support/resolved', 'type': 'support', 'category': 'support_touchpoints'},
            'feedback_survey': {'name': 'Feedback Survey', 'url': '/support/feedback', 'type': 'support', 'category': 'support_touchpoints'},
            
            # Mobile Specific
            'mobile_dashboard': {'name': 'Mobile Dashboard', 'url': '/mobile/dashboard', 'type': 'mobile', 'category': 'mobile_specific'},
            'quick_balance_check': {'name': 'Quick Balance', 'url': '/mobile/balance', 'type': 'mobile', 'category': 'mobile_specific'},
            'mobile_deposit_camera': {'name': 'Mobile Deposit', 'url': '/mobile/deposit', 'type': 'mobile', 'category': 'mobile_specific'},
            'location_services_enable': {'name': 'Enable Location', 'url': '/mobile/location', 'type': 'mobile', 'category': 'mobile_specific'},
            'push_notification_interaction': {'name': 'Push Notification', 'url': '/mobile/notifications', 'type': 'mobile', 'category': 'mobile_specific'},
            'app_settings_access': {'name': 'App Settings', 'url': '/mobile/settings', 'type': 'mobile', 'category': 'mobile_specific'},
            'biometric_setup': {'name': 'Biometric Setup', 'url': '/mobile/biometric', 'type': 'mobile', 'category': 'mobile_specific'},
            
            # Security Actions
            'security_center_visit': {'name': 'Security Center', 'url': '/security', 'type': 'security', 'category': 'security_actions'},
            'fraud_alert_review': {'name': 'Fraud Alerts', 'url': '/security/fraud', 'type': 'security', 'category': 'security_actions'},
            'card_management': {'name': 'Card Management', 'url': '/security/cards', 'type': 'security', 'category': 'security_actions'},
            'travel_notification_setup': {'name': 'Travel Notification', 'url': '/security/travel', 'type': 'security', 'category': 'security_actions'},
            'security_settings_update': {'name': 'Security Settings', 'url': '/security/settings', 'type': 'security', 'category': 'security_actions'},
            'device_management': {'name': 'Device Management', 'url': '/security/devices', 'type': 'security', 'category': 'security_actions'},
            'suspicious_activity_review': {'name': 'Suspicious Activity', 'url': '/security/suspicious', 'type': 'security', 'category': 'security_actions'},
            
            # Cross-selling
            'product_recommendation_view': {'name': 'Product Recommendations', 'url': '/recommendations', 'type': 'marketing', 'category': 'cross_selling'},
            'promotional_banner_click': {'name': 'Promotional Banner', 'url': '/promotions', 'type': 'marketing', 'category': 'cross_selling'},
            'upgrade_offer_consideration': {'name': 'Upgrade Offer', 'url': '/upgrade', 'type': 'marketing', 'category': 'cross_selling'},
            'additional_product_research': {'name': 'Additional Products', 'url': '/products/additional', 'type': 'marketing', 'category': 'cross_selling'},
            'cross_sell_application_start': {'name': 'Cross-sell Application', 'url': '/apply/cross-sell', 'type': 'application', 'category': 'cross_selling'},
            
            # Exits
            'logout': {'name': 'Logout', 'url': '/logout', 'type': 'authentication', 'category': 'exits'},
            'session_timeout': {'name': 'Session Timeout', 'url': '/timeout', 'type': 'system', 'category': 'exits'},
            'navigation_away': {'name': 'Navigate Away', 'url': '/external', 'type': 'system', 'category': 'exits'},
            'app_background': {'name': 'App Background', 'url': '/mobile/background', 'type': 'mobile', 'category': 'exits'},
            'browser_close': {'name': 'Browser Close', 'url': '/close', 'type': 'system', 'category': 'exits'},
            'phone_call_transfer': {'name': 'Phone Transfer', 'url': '/phone-transfer', 'type': 'support', 'category': 'exits'}
        }
        
        # Banking products and categories
        account_types = [
            'checking', 'savings', 'money_market', 'cd', 'credit_card', 
            'mortgage', 'auto_loan', 'personal_loan', 'heloc', 'investment_account',
            'business_checking', 'business_savings', 'business_loan'
        ]
        
        product_categories = [
            'deposit_accounts', 'credit_products', 'lending', 'investment_services',
            'insurance', 'wealth_management', 'business_banking', 'digital_services'
        ]
        
        customer_segments = [
            'mass_market', 'emerging_affluent', 'affluent', 'high_net_worth', 
            'ultra_high_net_worth', 'small_business', 'commercial', 'student',
            'senior', 'military'
        ]
        
        channels = ['web_desktop', 'web_mobile', 'mobile_app', 'tablet_app']
        
        # Technical configurations
        browsers = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Mobile Safari', 'Chrome Mobile', 'Samsung Internet']
        browser_versions = ['120.0', '119.0', '118.0', '117.0', '116.0', '115.0']
        operating_systems = [
            'Windows 10', 'Windows 11', 'macOS 14', 'macOS 13', 'macOS 12',
            'iOS 17', 'iOS 16', 'iOS 15', 'Android 14', 'Android 13', 'Android 12'
        ]
        device_types = ['Desktop', 'Mobile', 'Tablet']
        screen_resolutions = [
            '1920x1080', '1366x768', '1440x900', '2560x1440', '3840x2160',
            '375x667', '414x896', '390x844', '428x926',  # iPhone
            '1024x768', '1366x1024', '2048x2732'  # iPad
        ]
        
        # Campaign sources
        traffic_sources = [
            'direct', 'google', 'facebook', 'email', 'referral', 'paid_search',
            'youtube', 'linkedin', 'twitter', 'instagram'
        ]
        mediums = ['organic', 'cpc', 'email', 'social', 'referral', 'direct', 'display', 'video', 'affiliate']
        
        # Authentication methods
        auth_methods = [
            'username_password', 'biometric_fingerprint', 'biometric_face_id', 
            'sms_otp', 'email_otp', 'hardware_token', 'push_notification',
            'security_questions', 'voice_recognition'
        ]
        
        # Generate consistent user profile for this journey
        visitor_id = str(uuid.uuid4())
        customer_id = str(uuid.uuid4())
        customer_segment = random.choice(customer_segments)
        tenure_months = random.randint(1, 240)
        balance_tier = random.choices(
            ['low', 'medium', 'high', 'premium', 'private'],
            weights=[40, 30, 20, 8, 2]
        )[0]
        
        # Consistent geographic data
        state = fake.state()
        city = fake.city()
        zip_code = fake.zipcode()
        ip_address = fake.ipv4()  # Same IP for the journey
        
        # Choose a journey template
        journey_name = random.choice(list(journey_templates.keys()))
        journey_template = journey_templates[journey_name]
        
        # Build the actual event sequence from the template
        event_sequence = []
        for event_category, count in journey_template['base_flow']:
            selected_events = random.sample(shared_events[event_category], min(count, len(shared_events[event_category])))
            event_sequence.extend(selected_events)
        
        # Add some randomization - 20% chance to add extra cross-category events
        if random.random() < 0.2:
            extra_categories = [cat for cat in shared_events.keys() if cat not in ['exits']]
            extra_category = random.choice(extra_categories)
            extra_event = random.choice(shared_events[extra_category])
            # Insert at random position (not at the end)
            insert_pos = random.randint(1, len(event_sequence) - 1)
            event_sequence.insert(insert_pos, extra_event)
        
        # Consistent technical details for the journey
        device = random.choice(device_types)
        is_mobile = device in ['Mobile', 'Tablet']
        
        # Choose browser based on device
        if device == 'Mobile':
            browser = random.choice(['Mobile Safari', 'Chrome Mobile', 'Samsung Internet'])
            if browser == 'Mobile Safari':
                os = random.choice(['iOS 17', 'iOS 16', 'iOS 15'])
            else:
                os = random.choice(['Android 14', 'Android 13', 'Android 12'])
        elif device == 'Tablet':
            browser = random.choice(['Safari', 'Chrome', 'Mobile Safari'])
            os = random.choice(['iOS 17', 'iOS 16', 'macOS 14']) if 'Safari' in browser else random.choice(['Android 14', 'Windows 11'])
        else:  # Desktop
            browser = random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'])
            if browser == 'Safari':
                os = random.choice(['macOS 14', 'macOS 13', 'macOS 12'])
            else:
                os = random.choice(['Windows 11', 'Windows 10', 'macOS 14'])
        
        browser_version = random.choice(browser_versions)
        resolution = random.choice(screen_resolutions)
        user_agent = f"{browser}/{browser_version} ({os})"
        
        # Channel determination
        is_mobile_app = is_mobile and random.random() < 0.4
        if is_mobile_app:
            channel = 'mobile_app' if device == 'Mobile' else 'tablet_app'
        else:
            channel = f"web_{device.lower()}"
        
        # Campaign attribution (consistent for the journey)
        has_campaign = random.random() < 0.35
        campaign_id = str(uuid.uuid4()) if has_campaign else None
        traffic_source = random.choice(traffic_sources) if has_campaign else 'direct'
        medium = random.choice(mediums) if has_campaign else 'direct'
        referrer_domain = fake.domain_name() if traffic_source == 'referral' else None
        
        # Authentication method (consistent for journey)
        auth_method = random.choice(auth_methods)
        
        # Account type and product category
        account_type = random.choice(account_types)
        product_category = random.choice(product_categories)
        
        # Custom dimensions with explicit names (consistent for journey)
        customer_tenure = f"{tenure_months}_months"
        account_balance_tier = balance_tier
        product_interest = journey_template['primary_goal']
        mobile_app_version = f"v{random.randint(1, 15)}.{random.randint(0, 9)}.{random.randint(0, 9)}" if is_mobile_app else None
        
        clv_tiers = ['low', 'medium', 'high', 'premium']
        clv_weights = [40, 35, 20, 5] if balance_tier == 'low' else [10, 30, 40, 20]
        customer_lifetime_value_tier = random.choices(clv_tiers, weights=clv_weights)[0]
        
        # Generate journey start time
        journey_start = datetime.now() - timedelta(
            days=random.randint(0, 90),
            hours=random.randint(6, 23),
            minutes=random.randint(0, 59)
        )
        
        # Site section mapping
        site_section_mapping = {
            'authentication': 'Login & Security',
            'account_management': 'My Accounts',
            'marketing': 'Products & Services',
            'tools': 'Financial Tools',
            'application': 'Applications',
            'support': 'Customer Service',
            'transaction': 'Banking Services',
            'investment': 'Investments',
            'security': 'Security & Settings',
            'mobile': 'Mobile Banking',
            'legal': 'Legal & Compliance',
            'system': 'System'
        }
        
        # Generate events for the journey
        previous_url = None
        converted = False
        
        for i, event_type in enumerate(event_sequence):
            # Calculate event timestamp with more realistic gaps
            if i == 0:
                event_timestamp = journey_start
            else:
                # Variable time gaps - shorter for related actions, longer for different categories
                prev_category = event_details.get(event_sequence[i-1], {}).get('category', '')
                curr_category = event_details.get(event_type, {}).get('category', '')
                
                if prev_category == curr_category:
                    gap_seconds = random.randint(15, 120)  # 15 seconds to 2 minutes for related actions
                else:
                    gap_seconds = random.randint(60, 600)  # 1 to 10 minutes for category changes
                
                event_timestamp = previous_timestamp + timedelta(seconds=gap_seconds)
            
            previous_timestamp = event_timestamp
            
            # Get event information
            event_info = event_details.get(event_type, {
                'name': event_type.replace('_', ' ').title(),
                'url': f'/{event_type.replace("_", "-")}',
                'type': 'general',
                'category': 'other'
            })
            
            # Transaction amounts for specific events
            transaction_amount = None
            revenue_impact = None
            
            # Determine if this is a conversion event
            conversion_events = [
                'application_confirmation', 'trade_execution', 'payment_confirmation',
                'resolution_confirmation', 'account_closed_confirmation'
            ]
            
            is_conversion_event = event_type in conversion_events
            
            if is_conversion_event and random.random() < journey_template['conversion_rate']:
                converted = True
                if journey_template['revenue_range'][1] > 0:
                    revenue_impact = round(random.uniform(*journey_template['revenue_range']), 2)
                else:
                    revenue_impact = journey_template['revenue_range'][0]  # Negative or zero
            
            # Transaction amounts based on event type
            if 'payment' in event_type or 'transfer' in event_type:
                transaction_amount = round(random.uniform(25, 5000), 2)
            elif 'trade' in event_type:
                transaction_amount = round(random.uniform(500, 100000), 2)
            
            # Custom events with explicit names (counts)
            form_starts = 1 if any(x in event_type for x in ['start', 'initiate', 'begin']) else 0
            form_completions = 1 if any(x in event_type for x in ['confirmation', 'submit', 'complete', 'success', 'execution']) else 0
            errors_encountered = 1 if random.random() < 0.03 else 0  # 3% error rate
            support_interactions = 1 if event_info['category'] == 'support_touchpoints' else 0
            product_views = 1 if event_info['category'] in ['product_research', 'cross_selling'] else 0
            
            # Page interaction metrics
            time_on_page = random.randint(10, 900)  # 10 seconds to 15 minutes
            scroll_depth = random.randint(5, 100)  # Percentage
            clicks_on_page = random.randint(0, 20)
            page_load_time = random.randint(150, 5000)  # milliseconds
            
            yield (
                # Core identifiers
                str(uuid.uuid4()),  # event_id
                visitor_id,  # visitor_id (consistent)
                customer_id,  # customer_id (consistent)
                
                # Event details
                event_timestamp,  # event_timestamp
                event_type,  # event_type
                event_info['category'],  # event_category
                event_type.replace('_', ' ').title(),  # event_action
                f"{journey_template['primary_goal']}_{event_type}",  # event_label
                
                # Page information
                event_info['name'],  # page_name
                event_info['url'],  # page_url
                event_info['type'],  # page_type
                site_section_mapping.get(event_info['type'], 'Other'),  # site_section
                previous_url,  # referrer_url
                
                # Technical details (consistent for journey)
                browser,  # browser
                browser_version,  # browser_version
                os,  # operating_system
                device,  # device_type
                resolution,  # screen_resolution
                user_agent,  # user_agent
                ip_address,  # ip_address
                
                # Geographic data (consistent for journey)
                'United States',  # country
                state,  # state
                city,  # city
                zip_code,  # zip_code
                
                # Page interaction details
                time_on_page,  # time_on_page
                scroll_depth,  # scroll_depth
                clicks_on_page,  # clicks_on_page
                
                # Banking specific fields
                account_type,  # account_type
                product_category,  # product_category
                transaction_amount,  # transaction_amount
                channel,  # channel
                auth_method,  # authentication_method
                customer_segment,  # customer_segment
                
                # Campaign/Marketing (consistent for journey)
                campaign_id,  # campaign_id
                traffic_source,  # traffic_source
                medium,  # medium
                referrer_domain,  # referrer_domain
                
                # Custom dimensions with explicit names (consistent for journey)
                customer_tenure,  # customer_tenure
                account_balance_tier,  # account_balance_tier
                product_interest,  # product_interest
                mobile_app_version,  # mobile_app_version
                customer_lifetime_value_tier,  # customer_lifetime_value_tier
                
                # Custom events with explicit names
                form_starts,  # form_starts
                form_completions,  # form_completions
                errors_encountered,  # errors_encountered
                support_interactions,  # support_interactions
                product_views,  # product_views
                
                # Additional context
                is_mobile_app,  # is_mobile_app
                page_load_time,  # page_load_time_ms
                is_conversion_event and converted,  # conversion_flag
                revenue_impact  # revenue_impact
            )
            
            # Set previous URL for next iteration
            previous_url = event_info['url']
$$;

select * from bank_event_stream;