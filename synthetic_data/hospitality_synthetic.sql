CREATE OR REPLACE FUNCTION generate_hotel_journey()
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
    
    -- Hotel/Travel specific fields
    hotel_name STRING,
    hotel_brand STRING,
    destination_city STRING,
    destination_country STRING,
    room_type STRING,
    rate_plan STRING,
    check_in_date DATE,
    check_out_date DATE,
    nights_stay INT,
    guests_count INT,
    room_rate DECIMAL(10,2),
    total_booking_value DECIMAL(12,2),
    currency_code STRING,
    
    -- Campaign/Marketing (Adobe Analytics style)
    campaign_id STRING,
    traffic_source STRING,
    medium STRING,
    referrer_domain STRING,
    
    -- Custom dimensions with explicit names
    traveler_type STRING,
    booking_purpose STRING,
    loyalty_tier STRING,
    advance_booking_days INT,
    price_sensitivity STRING,
    
    -- Custom events with explicit names
    hotel_searches INT,
    property_views INT,
    booking_starts INT,
    booking_completions INT,
    cancellation_requests INT,
    
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
from datetime import datetime, timedelta, date
from faker import Faker

fake = Faker()

class generateJourney:
    def process(self):
        # Define shared event pools for hotel/travel booking
        shared_events = {
            'entry_points': [
                'homepage_visit', 'destination_landing', 'search_result_click', 'email_campaign_click',
                'social_media_click', 'mobile_app_open', 'travel_blog_referral', 'google_ads_click',
                'ota_comparison_click', 'direct_url_entry', 'metasearch_referral'
            ],
            'authentication': [
                'login_attempt', 'login_success', 'guest_booking_start', 'account_creation_start',
                'password_reset_request', 'social_login_attempt', 'loyalty_login', 'corporate_login'
            ],
            'search_discovery': [
                'destination_search', 'date_selection', 'guest_count_selection', 'search_execution',
                'search_refinement', 'filter_application', 'sort_by_price', 'sort_by_rating',
                'map_view_toggle', 'list_view_toggle', 'availability_check'
            ],
            'property_research': [
                'hotel_detail_view', 'photo_gallery_view', 'amenities_section_view', 'location_map_view',
                'reviews_section_view', 'room_types_comparison', 'rate_calendar_view', 'virtual_tour_start',
                'nearby_attractions_view', 'hotel_policies_view', 'cancellation_policy_view'
            ],
            'booking_process': [
                'room_selection', 'rate_plan_selection', 'guest_info_entry', 'special_requests_entry',
                'extras_selection', 'payment_info_entry', 'booking_review', 'terms_acceptance',
                'booking_confirmation', 'confirmation_email_sent'
            ],
            'account_management': [
                'my_trips_view', 'booking_history_view', 'profile_update', 'preferences_update',
                'payment_methods_management', 'loyalty_account_view', 'points_balance_check',
                'membership_benefits_view', 'communication_preferences'
            ],
            'trip_planning': [
                'itinerary_builder', 'destination_guide_view', 'weather_check', 'flight_search',
                'car_rental_search', 'activity_booking', 'restaurant_reservations', 'travel_insurance_view',
                'packing_list_creation', 'travel_tips_view'
            ],
            'loyalty_rewards': [
                'loyalty_program_join', 'points_earning_view', 'points_redemption', 'tier_benefits_view',
                'elite_status_check', 'bonus_points_offers', 'partner_offers_view', 'reward_nights_booking'
            ],
            'reviews_feedback': [
                'review_submission', 'review_reading', 'rating_submission', 'photo_review_upload',
                'experience_sharing', 'recommendation_writing', 'complaint_submission', 'feedback_survey'
            ],
            'support_touchpoints': [
                'help_center_visit', 'faq_browse', 'live_chat_initiate', 'phone_support_request',
                'booking_modification_request', 'cancellation_request', 'refund_inquiry',
                'special_assistance_request', 'group_booking_inquiry', 'concierge_service_request'
            ],
            'mobile_specific': [
                'mobile_check_in', 'digital_key_setup', 'room_service_order', 'housekeeping_request',
                'wake_up_call_setup', 'mobile_checkout', 'push_notification_interaction',
                'location_services_enable', 'offline_itinerary_access'
            ],
            'promotional': [
                'deal_alerts_signup', 'flash_sale_participation', 'package_deal_view', 'group_discount_inquiry',
                'corporate_rate_access', 'promo_code_entry', 'last_minute_deals_view', 'seasonal_promotion_view',
                'loyalty_bonus_activation', 'referral_program_use'
            ],
            'comparison_shopping': [
                'rate_comparison_view', 'amenities_comparison', 'location_comparison', 'review_score_comparison',
                'price_alert_setup', 'competitor_rate_check', 'value_for_money_analysis', 'alternative_dates_check'
            ],
            'exits': [
                'logout', 'session_timeout', 'navigation_away', 'app_background',
                'browser_close', 'booking_abandonment', 'search_abandonment'
            ]
        }
        
        # Define journey templates for hotel booking
        journey_templates = {
            'leisure_vacation_booking': {
                'primary_goal': 'vacation_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('search_discovery', random.randint(2, 4)),
                    ('property_research', random.randint(3, 6)),
                    ('comparison_shopping', random.randint(1, 3)),
                    ('trip_planning', random.randint(0, 2)),
                    ('booking_process', random.randint(4, 8)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.18,
                'revenue_range': (200, 1500)
            },
            'business_travel_booking': {
                'primary_goal': 'business_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('search_discovery', random.randint(1, 3)),
                    ('property_research', random.randint(1, 3)),
                    ('booking_process', random.randint(3, 6)),
                    ('account_management', random.randint(0, 1)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.35,
                'revenue_range': (150, 800)
            },
            'last_minute_booking': {
                'primary_goal': 'urgent_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('promotional', random.randint(1, 2)),
                    ('search_discovery', random.randint(1, 2)),
                    ('property_research', random.randint(1, 3)),
                    ('booking_process', random.randint(3, 5)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.45,
                'revenue_range': (100, 600)
            },
            'research_planning_phase': {
                'primary_goal': 'travel_research',
                'base_flow': [
                    ('entry_points', 1),
                    ('search_discovery', random.randint(3, 6)),
                    ('property_research', random.randint(4, 8)),
                    ('trip_planning', random.randint(2, 4)),
                    ('comparison_shopping', random.randint(2, 4)),
                    ('support_touchpoints', random.randint(0, 2)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.05,
                'revenue_range': (0, 0)
            },
            'loyalty_member_booking': {
                'primary_goal': 'loyalty_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('loyalty_rewards', random.randint(1, 3)),
                    ('search_discovery', random.randint(1, 3)),
                    ('property_research', random.randint(2, 4)),
                    ('booking_process', random.randint(3, 6)),
                    ('account_management', random.randint(0, 1)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.42,
                'revenue_range': (180, 1200)
            },
            'group_event_booking': {
                'primary_goal': 'group_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('search_discovery', random.randint(2, 4)),
                    ('property_research', random.randint(3, 5)),
                    ('support_touchpoints', random.randint(2, 4)),
                    ('comparison_shopping', random.randint(1, 3)),
                    ('booking_process', random.randint(4, 7)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.25,
                'revenue_range': (800, 5000)
            },
            'mobile_app_browsing': {
                'primary_goal': 'mobile_engagement',
                'base_flow': [
                    ('entry_points', 1),
                    ('mobile_specific', random.randint(2, 4)),
                    ('search_discovery', random.randint(2, 4)),
                    ('property_research', random.randint(1, 3)),
                    ('booking_process', random.randint(0, 4)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.15,
                'revenue_range': (120, 700)
            },
            'customer_service_interaction': {
                'primary_goal': 'service_request',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(0, 1)),
                    ('account_management', random.randint(1, 2)),
                    ('support_touchpoints', random.randint(3, 6)),
                    ('booking_process', random.randint(0, 3)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.30,
                'revenue_range': (0, 200)
            },
            'booking_modification': {
                'primary_goal': 'change_booking',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('account_management', random.randint(1, 2)),
                    ('support_touchpoints', random.randint(1, 3)),
                    ('search_discovery', random.randint(0, 2)),
                    ('booking_process', random.randint(0, 4)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.55,
                'revenue_range': (-100, 300)
            },
            'price_comparison_shopping': {
                'primary_goal': 'price_research',
                'base_flow': [
                    ('entry_points', 1),
                    ('search_discovery', random.randint(2, 4)),
                    ('comparison_shopping', random.randint(3, 6)),
                    ('property_research', random.randint(2, 5)),
                    ('promotional', random.randint(1, 2)),
                    ('booking_process', random.randint(0, 3)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.12,
                'revenue_range': (100, 800)
            }
        }
        
        # Detailed event mappings for hotel/travel booking
        event_details = {
            # Entry points
            'homepage_visit': {'name': 'Homepage', 'url': '/', 'type': 'marketing', 'category': 'navigation'},
            'destination_landing': {'name': 'Destination Landing', 'url': '/destinations/paris', 'type': 'marketing', 'category': 'destination'},
            'search_result_click': {'name': 'Search Results', 'url': '/search-results', 'type': 'search', 'category': 'search'},
            'email_campaign_click': {'name': 'Email Campaign', 'url': '/campaign/summer-deals', 'type': 'marketing', 'category': 'campaign'},
            'social_media_click': {'name': 'Social Media', 'url': '/social-landing', 'type': 'marketing', 'category': 'social'},
            'mobile_app_open': {'name': 'Mobile App Home', 'url': '/app/home', 'type': 'mobile', 'category': 'mobile'},
            'travel_blog_referral': {'name': 'Travel Blog', 'url': '/blog-referral', 'type': 'marketing', 'category': 'content'},
            'google_ads_click': {'name': 'Google Ads', 'url': '/ads-landing', 'type': 'marketing', 'category': 'paid_search'},
            'ota_comparison_click': {'name': 'OTA Comparison', 'url': '/ota-referral', 'type': 'marketing', 'category': 'comparison'},
            'direct_url_entry': {'name': 'Direct Entry', 'url': '/direct', 'type': 'direct', 'category': 'direct'},
            'metasearch_referral': {'name': 'Metasearch', 'url': '/metasearch-referral', 'type': 'marketing', 'category': 'metasearch'},
            
            # Authentication
            'login_attempt': {'name': 'Login Page', 'url': '/login', 'type': 'authentication', 'category': 'auth'},
            'login_success': {'name': 'Login Success', 'url': '/my-account', 'type': 'authentication', 'category': 'auth'},
            'guest_booking_start': {'name': 'Guest Booking', 'url': '/book-as-guest', 'type': 'booking', 'category': 'auth'},
            'account_creation_start': {'name': 'Create Account', 'url': '/register', 'type': 'authentication', 'category': 'auth'},
            'password_reset_request': {'name': 'Password Reset', 'url': '/forgot-password', 'type': 'authentication', 'category': 'auth'},
            'social_login_attempt': {'name': 'Social Login', 'url': '/login/social', 'type': 'authentication', 'category': 'auth'},
            'loyalty_login': {'name': 'Loyalty Login', 'url': '/loyalty/login', 'type': 'authentication', 'category': 'loyalty'},
            'corporate_login': {'name': 'Corporate Login', 'url': '/corporate/login', 'type': 'authentication', 'category': 'corporate'},
            
            # Search & Discovery
            'destination_search': {'name': 'Destination Search', 'url': '/search', 'type': 'search', 'category': 'search'},
            'date_selection': {'name': 'Date Selection', 'url': '/search/dates', 'type': 'search', 'category': 'search'},
            'guest_count_selection': {'name': 'Guest Count', 'url': '/search/guests', 'type': 'search', 'category': 'search'},
            'search_execution': {'name': 'Execute Search', 'url': '/search/results', 'type': 'search', 'category': 'search'},
            'search_refinement': {'name': 'Refine Search', 'url': '/search/refine', 'type': 'search', 'category': 'search'},
            'filter_application': {'name': 'Apply Filters', 'url': '/search/filters', 'type': 'search', 'category': 'filter'},
            'sort_by_price': {'name': 'Sort by Price', 'url': '/search/sort-price', 'type': 'search', 'category': 'sort'},
            'sort_by_rating': {'name': 'Sort by Rating', 'url': '/search/sort-rating', 'type': 'search', 'category': 'sort'},
            'map_view_toggle': {'name': 'Map View', 'url': '/search/map', 'type': 'search', 'category': 'view'},
            'list_view_toggle': {'name': 'List View', 'url': '/search/list', 'type': 'search', 'category': 'view'},
            'availability_check': {'name': 'Check Availability', 'url': '/search/availability', 'type': 'search', 'category': 'availability'},
            
            # Property Research
            'hotel_detail_view': {'name': 'Hotel Details', 'url': '/hotel/grand-plaza-paris', 'type': 'property', 'category': 'property'},
            'photo_gallery_view': {'name': 'Photo Gallery', 'url': '/hotel/grand-plaza-paris/photos', 'type': 'property', 'category': 'media'},
            'amenities_section_view': {'name': 'Hotel Amenities', 'url': '/hotel/grand-plaza-paris/amenities', 'type': 'property', 'category': 'amenities'},
            'location_map_view': {'name': 'Location Map', 'url': '/hotel/grand-plaza-paris/location', 'type': 'property', 'category': 'location'},
            'reviews_section_view': {'name': 'Guest Reviews', 'url': '/hotel/grand-plaza-paris/reviews', 'type': 'property', 'category': 'reviews'},
            'room_types_comparison': {'name': 'Room Types', 'url': '/hotel/grand-plaza-paris/rooms', 'type': 'property', 'category': 'rooms'},
            'rate_calendar_view': {'name': 'Rate Calendar', 'url': '/hotel/grand-plaza-paris/rates', 'type': 'property', 'category': 'pricing'},
            'virtual_tour_start': {'name': 'Virtual Tour', 'url': '/hotel/grand-plaza-paris/tour', 'type': 'property', 'category': 'media'},
            'nearby_attractions_view': {'name': 'Nearby Attractions', 'url': '/hotel/grand-plaza-paris/attractions', 'type': 'property', 'category': 'location'},
            'hotel_policies_view': {'name': 'Hotel Policies', 'url': '/hotel/grand-plaza-paris/policies', 'type': 'property', 'category': 'policies'},
            'cancellation_policy_view': {'name': 'Cancellation Policy', 'url': '/hotel/grand-plaza-paris/cancellation', 'type': 'property', 'category': 'policies'},
            
            # Booking Process
            'room_selection': {'name': 'Select Room', 'url': '/book/room-selection', 'type': 'booking', 'category': 'booking'},
            'rate_plan_selection': {'name': 'Select Rate Plan', 'url': '/book/rate-plan', 'type': 'booking', 'category': 'booking'},
            'guest_info_entry': {'name': 'Guest Information', 'url': '/book/guest-info', 'type': 'booking', 'category': 'booking'},
            'special_requests_entry': {'name': 'Special Requests', 'url': '/book/special-requests', 'type': 'booking', 'category': 'booking'},
            'extras_selection': {'name': 'Select Extras', 'url': '/book/extras', 'type': 'booking', 'category': 'booking'},
            'payment_info_entry': {'name': 'Payment Information', 'url': '/book/payment', 'type': 'booking', 'category': 'payment'},
            'booking_review': {'name': 'Review Booking', 'url': '/book/review', 'type': 'booking', 'category': 'booking'},
            'terms_acceptance': {'name': 'Accept Terms', 'url': '/book/terms', 'type': 'booking', 'category': 'legal'},
            'booking_confirmation': {'name': 'Booking Confirmed', 'url': '/book/confirmation', 'type': 'booking', 'category': 'confirmation'},
            'confirmation_email_sent': {'name': 'Confirmation Email', 'url': '/email/confirmation', 'type': 'email', 'category': 'confirmation'},
            
            # Account Management
            'my_trips_view': {'name': 'My Trips', 'url': '/my-account/trips', 'type': 'account', 'category': 'account'},
            'booking_history_view': {'name': 'Booking History', 'url': '/my-account/history', 'type': 'account', 'category': 'account'},
            'profile_update': {'name': 'Update Profile', 'url': '/my-account/profile', 'type': 'account', 'category': 'account'},
            'preferences_update': {'name': 'Travel Preferences', 'url': '/my-account/preferences', 'type': 'account', 'category': 'preferences'},
            'payment_methods_management': {'name': 'Payment Methods', 'url': '/my-account/payments', 'type': 'account', 'category': 'payment'},
            'loyalty_account_view': {'name': 'Loyalty Account', 'url': '/loyalty/account', 'type': 'account', 'category': 'loyalty'},
            'points_balance_check': {'name': 'Points Balance', 'url': '/loyalty/points', 'type': 'account', 'category': 'loyalty'},
            'membership_benefits_view': {'name': 'Member Benefits', 'url': '/loyalty/benefits', 'type': 'account', 'category': 'loyalty'},
            'communication_preferences': {'name': 'Communication Prefs', 'url': '/my-account/communications', 'type': 'account', 'category': 'preferences'},
            
            # Trip Planning
            'itinerary_builder': {'name': 'Itinerary Builder', 'url': '/trip-planner', 'type': 'planning', 'category': 'planning'},
            'destination_guide_view': {'name': 'Destination Guide', 'url': '/guides/paris', 'type': 'planning', 'category': 'destination'},
            'weather_check': {'name': 'Weather Forecast', 'url': '/weather/paris', 'type': 'planning', 'category': 'weather'},
            'flight_search': {'name': 'Flight Search', 'url': '/flights', 'type': 'planning', 'category': 'flights'},
            'car_rental_search': {'name': 'Car Rental', 'url': '/cars', 'type': 'planning', 'category': 'transport'},
            'activity_booking': {'name': 'Activity Booking', 'url': '/activities', 'type': 'planning', 'category': 'activities'},
            'restaurant_reservations': {'name': 'Restaurant Reservations', 'url': '/restaurants', 'type': 'planning', 'category': 'dining'},
            'travel_insurance_view': {'name': 'Travel Insurance', 'url': '/insurance', 'type': 'planning', 'category': 'insurance'},
            'packing_list_creation': {'name': 'Packing List', 'url': '/packing-list', 'type': 'planning', 'category': 'preparation'},
            'travel_tips_view': {'name': 'Travel Tips', 'url': '/tips', 'type': 'planning', 'category': 'tips'},
            
            # Loyalty & Rewards
            'loyalty_program_join': {'name': 'Join Loyalty Program', 'url': '/loyalty/join', 'type': 'loyalty', 'category': 'loyalty'},
            'points_earning_view': {'name': 'Earn Points', 'url': '/loyalty/earn', 'type': 'loyalty', 'category': 'loyalty'},
            'points_redemption': {'name': 'Redeem Points', 'url': '/loyalty/redeem', 'type': 'loyalty', 'category': 'redemption'},
            'tier_benefits_view': {'name': 'Tier Benefits', 'url': '/loyalty/tiers', 'type': 'loyalty', 'category': 'benefits'},
            'elite_status_check': {'name': 'Elite Status', 'url': '/loyalty/elite', 'type': 'loyalty', 'category': 'status'},
            'bonus_points_offers': {'name': 'Bonus Points Offers', 'url': '/loyalty/bonus', 'type': 'loyalty', 'category': 'offers'},
            'partner_offers_view': {'name': 'Partner Offers', 'url': '/loyalty/partners', 'type': 'loyalty', 'category': 'partners'},
            'reward_nights_booking': {'name': 'Reward Nights', 'url': '/loyalty/reward-nights', 'type': 'loyalty', 'category': 'rewards'},
            
            # Reviews & Feedback
            'review_submission': {'name': 'Submit Review', 'url': '/review/submit', 'type': 'social', 'category': 'review'},
            'review_reading': {'name': 'Read Reviews', 'url': '/reviews', 'type': 'social', 'category': 'social_proof'},
            'rating_submission': {'name': 'Submit Rating', 'url': '/rating/submit', 'type': 'social', 'category': 'rating'},
            'photo_review_upload': {'name': 'Photo Review', 'url': '/review/photos', 'type': 'social', 'category': 'ugc'},
            'experience_sharing': {'name': 'Share Experience', 'url': '/share-experience', 'type': 'social', 'category': 'sharing'},
            'recommendation_writing': {'name': 'Write Recommendation', 'url': '/recommend', 'type': 'social', 'category': 'recommendation'},
            'complaint_submission': {'name': 'Submit Complaint', 'url': '/complaint', 'type': 'support', 'category': 'complaint'},
            'feedback_survey': {'name': 'Feedback Survey', 'url': '/survey', 'type': 'feedback', 'category': 'survey'},
            
            # Support Touchpoints
            'help_center_visit': {'name': 'Help Center', 'url': '/help', 'type': 'support', 'category': 'support'},
            'faq_browse': {'name': 'FAQ', 'url': '/faq', 'type': 'support', 'category': 'support'},
            'live_chat_initiate': {'name': 'Live Chat', 'url': '/chat', 'type': 'support', 'category': 'support'},
            'phone_support_request': {'name': 'Phone Support', 'url': '/support/phone', 'type': 'support', 'category': 'phone'},
            'booking_modification_request': {'name': 'Modify Booking', 'url': '/modify-booking', 'type': 'support', 'category': 'modification'},
            'cancellation_request': {'name': 'Cancel Booking', 'url': '/cancel-booking', 'type': 'support', 'category': 'cancellation'},
            'refund_inquiry': {'name': 'Refund Inquiry', 'url': '/refund', 'type': 'support', 'category': 'refund'},
            'special_assistance_request': {'name': 'Special Assistance', 'url': '/special-assistance', 'type': 'support', 'category': 'assistance'},
            'group_booking_inquiry': {'name': 'Group Booking', 'url': '/group-booking', 'type': 'support', 'category': 'group'},
            'concierge_service_request': {'name': 'Concierge Service', 'url': '/concierge', 'type': 'support', 'category': 'concierge'},
            
            # Mobile Specific
            'mobile_check_in': {'name': 'Mobile Check-in', 'url': '/app/check-in', 'type': 'mobile', 'category': 'checkin'},
            'digital_key_setup': {'name': 'Digital Key', 'url': '/app/digital-key', 'type': 'mobile', 'category': 'key'},
            'room_service_order': {'name': 'Room Service', 'url': '/app/room-service', 'type': 'mobile', 'category': 'service'},
            'housekeeping_request': {'name': 'Housekeeping Request', 'url': '/app/housekeeping', 'type': 'mobile', 'category': 'service'},
            'wake_up_call_setup': {'name': 'Wake-up Call', 'url': '/app/wake-up', 'type': 'mobile', 'category': 'service'},
            'mobile_checkout': {'name': 'Mobile Checkout', 'url': '/app/checkout', 'type': 'mobile', 'category': 'checkout'},
            'push_notification_interaction': {'name': 'Push Notification', 'url': '/app/notification', 'type': 'mobile', 'category': 'notification'},
            'location_services_enable': {'name': 'Location Services', 'url': '/app/location', 'type': 'mobile', 'category': 'location'},
            'offline_itinerary_access': {'name': 'Offline Itinerary', 'url': '/app/offline', 'type': 'mobile', 'category': 'offline'},
            
            # Promotional
            'deal_alerts_signup': {'name': 'Deal Alerts', 'url': '/deals/alerts', 'type': 'promotion', 'category': 'alerts'},
            'flash_sale_participation': {'name': 'Flash Sale', 'url': '/flash-sale', 'type': 'promotion', 'category': 'flash_sale'},
            'package_deal_view': {'name': 'Package Deals', 'url': '/packages', 'type': 'promotion', 'category': 'packages'},
            'group_discount_inquiry': {'name': 'Group Discounts', 'url': '/group-discounts', 'type': 'promotion', 'category': 'group'},
            'corporate_rate_access': {'name': 'Corporate Rates', 'url': '/corporate-rates', 'type': 'promotion', 'category': 'corporate'},
            'promo_code_entry': {'name': 'Promo Code', 'url': '/promo-code', 'type': 'promotion', 'category': 'promo'},
            'last_minute_deals_view': {'name': 'Last Minute Deals', 'url': '/last-minute', 'type': 'promotion', 'category': 'last_minute'},
            'seasonal_promotion_view': {'name': 'Seasonal Promotion', 'url': '/seasonal-deals', 'type': 'promotion', 'category': 'seasonal'},
            'loyalty_bonus_activation': {'name': 'Loyalty Bonus', 'url': '/loyalty/bonus-activation', 'type': 'promotion', 'category': 'loyalty_bonus'},
            'referral_program_use': {'name': 'Referral Program', 'url': '/referral', 'type': 'promotion', 'category': 'referral'},
            
            # Comparison Shopping
            'rate_comparison_view': {'name': 'Rate Comparison', 'url': '/compare/rates', 'type': 'comparison', 'category': 'pricing'},
            'amenities_comparison': {'name': 'Amenities Comparison', 'url': '/compare/amenities', 'type': 'comparison', 'category': 'amenities'},
            'location_comparison': {'name': 'Location Comparison', 'url': '/compare/location', 'type': 'comparison', 'category': 'location'},
            'review_score_comparison': {'name': 'Review Comparison', 'url': '/compare/reviews', 'type': 'comparison', 'category': 'reviews'},
            'price_alert_setup': {'name': 'Price Alerts', 'url': '/price-alerts', 'type': 'comparison', 'category': 'alerts'},
            'competitor_rate_check': {'name': 'Competitor Rates', 'url': '/competitor-rates', 'type': 'comparison', 'category': 'competitive'},
            'value_for_money_analysis': {'name': 'Value Analysis', 'url': '/value-analysis', 'type': 'comparison', 'category': 'value'},
            'alternative_dates_check': {'name': 'Alternative Dates', 'url': '/alternative-dates', 'type': 'comparison', 'category': 'dates'},
            
            # Exits
            'logout': {'name': 'Logout', 'url': '/logout', 'type': 'authentication', 'category': 'exit'},
            'session_timeout': {'name': 'Session Timeout', 'url': '/timeout', 'type': 'system', 'category': 'exit'},
            'navigation_away': {'name': 'Navigate Away', 'url': '/external', 'type': 'system', 'category': 'exit'},
            'app_background': {'name': 'App Background', 'url': '/app/background', 'type': 'mobile', 'category': 'exit'},
            'browser_close': {'name': 'Browser Close', 'url': '/close', 'type': 'system', 'category': 'exit'},
            'booking_abandonment': {'name': 'Booking Abandonment', 'url': '/booking/abandon', 'type': 'booking', 'category': 'abandonment'},
            'search_abandonment': {'name': 'Search Abandonment', 'url': '/search/abandon', 'type': 'search', 'category': 'abandonment'}
        }
        
        # Hotel and travel data
        hotel_names = [
            'Grand Plaza Hotel', 'Marriott Downtown', 'Hilton Garden Inn', 'Holiday Inn Express',
            'Hyatt Regency', 'Sheraton Grand', 'Four Seasons Resort', 'Ritz-Carlton',
            'Hampton Inn & Suites', 'Courtyard by Marriott', 'DoubleTree by Hilton',
            'Embassy Suites', 'Westin Resort', 'W Hotel', 'Aloft Hotel',
            'Renaissance Hotel', 'JW Marriott', 'St. Regis Resort', 'Edition Hotel',
            'Waldorf Astoria', 'Conrad Hotel', 'InterContinental', 'Crowne Plaza',
            'Hotel Indigo', 'Kimpton Hotel'
        ]
        
        hotel_brands = [
            'Marriott', 'Hilton', 'Hyatt', 'IHG', 'Accor', 'Wyndham',
            'Choice Hotels', 'Best Western', 'Radisson', 'Independent'
        ]
        
        destinations = [
            ('New York', 'United States'), ('Paris', 'France'), ('London', 'United Kingdom'),
            ('Tokyo', 'Japan'), ('Dubai', 'UAE'), ('Los Angeles', 'United States'),
            ('Barcelona', 'Spain'), ('Rome', 'Italy'), ('Amsterdam', 'Netherlands'),
            ('Singapore', 'Singapore'), ('Sydney', 'Australia'), ('Miami', 'United States'),
            ('Las Vegas', 'United States'), ('Bangkok', 'Thailand'), ('Istanbul', 'Turkey'),
            ('Berlin', 'Germany'), ('Vienna', 'Austria'), ('Prague', 'Czech Republic')
        ]
        
        room_types = [
            'Standard King', 'Standard Queen', 'Deluxe King', 'Deluxe Queen',
            'Junior Suite', 'Executive Suite', 'Presidential Suite',
            'Standard Double', 'Superior King', 'Premium Queen'
        ]
        
        rate_plans = [
            'Best Available Rate', 'Advance Purchase', 'Stay Longer Save More',
            'Flexible Rate', 'Non-Refundable', 'Corporate Rate', 'AAA Rate',
            'Senior Rate', 'Government Rate', 'Package Deal'
        ]
        
        traveler_types = [
            'leisure_solo', 'leisure_couple', 'leisure_family', 'business_solo',
            'business_group', 'group_leisure', 'group_business', 'bleisure'
        ]
        
        booking_purposes = [
            'vacation', 'business_meeting', 'conference', 'wedding', 'family_visit',
            'romantic_getaway', 'adventure_travel', 'city_break', 'staycation', 'relocation'
        ]
        
        loyalty_tiers = ['None', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Titanium']
        price_sensitivities = ['budget', 'value', 'mid_range', 'luxury', 'ultra_luxury']
        
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
            'direct', 'google', 'facebook', 'instagram', 'email', 'referral', 'paid_search',
            'booking.com', 'expedia', 'tripadvisor', 'kayak', 'priceline', 'travel_blog'
        ]
        mediums = ['organic', 'cpc', 'email', 'social', 'referral', 'direct', 'display', 'video', 'ota', 'metasearch']
        
        # Generate consistent user profile for this journey
        visitor_id = str(uuid.uuid4())
        customer_id = str(uuid.uuid4())
        traveler_type = random.choice(traveler_types)
        booking_purpose = random.choice(booking_purposes)
        loyalty_tier = random.choice(loyalty_tiers)
        price_sensitivity = random.choice(price_sensitivities)
        
        # Consistent geographic data
        state = fake.state()
        city = fake.city()
        zip_code = fake.zipcode()
        ip_address = fake.ipv4()
        
        # Choose a journey template
        journey_name = random.choice(list(journey_templates.keys()))
        journey_template = journey_templates[journey_name]
        
        # Build the actual event sequence from the template
        event_sequence = []
        for event_category, count in journey_template['base_flow']:
            selected_events = random.sample(shared_events[event_category], min(count, len(shared_events[event_category])))
            event_sequence.extend(selected_events)
        
        # Add some randomization - 20% chance to add extra cross-category events
        if random.random() < 0.20:
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
        is_mobile_app = is_mobile and random.random() < 0.25
        if is_mobile_app:
            channel = 'mobile_app' if device == 'Mobile' else 'tablet_app'
        else:
            channel = f"web_{device.lower()}"
        
        # Campaign attribution (consistent for the journey)
        has_campaign = random.random() < 0.45
        campaign_id = str(uuid.uuid4()) if has_campaign else None
        traffic_source = random.choice(traffic_sources) if has_campaign else 'direct'
        medium = random.choice(mediums) if has_campaign else 'direct'
        referrer_domain = fake.domain_name() if traffic_source == 'referral' else None
        
        # Travel details (consistent for the journey)
        destination_city, destination_country = random.choice(destinations)
        hotel_name = random.choice(hotel_names)
        hotel_brand = random.choice(hotel_brands)
        room_type = random.choice(room_types)
        rate_plan = random.choice(rate_plans)
        
        # Generate realistic travel dates
        advance_days = random.randint(1, 120)  # 1 to 120 days in advance
        check_in_date = date.today() + timedelta(days=advance_days)
        nights_stay = random.randint(1, 14)  # 1 to 14 nights
        check_out_date = check_in_date + timedelta(days=nights_stay)
        guests_count = random.randint(1, 4)
        
        # Generate room rate based on destination and room type
        base_rates = {'Standard': 120, 'Deluxe': 180, 'Suite': 300}
        rate_multiplier = 1.0
        for rate_type in base_rates:
            if rate_type.lower() in room_type.lower():
                room_rate = base_rates[rate_type] * rate_multiplier
                break
        else:
            room_rate = 150
        
        room_rate = round(room_rate * random.uniform(0.7, 1.5), 2)  # Add variability
        
        # Generate journey start time
        journey_start = datetime.now() - timedelta(
            days=random.randint(0, 90),
            hours=random.randint(6, 23),
            minutes=random.randint(0, 59)
        )
        
        # Site section mapping
        site_section_mapping = {
            'marketing': 'Marketing & Deals',
            'authentication': 'Account & Login',
            'search': 'Search & Discovery',
            'property': 'Hotel Details',
            'booking': 'Booking & Reservations',
            'account': 'My Account',
            'planning': 'Trip Planning',
            'loyalty': 'Loyalty Program',
            'social': 'Reviews & Community',
            'support': 'Customer Support',
            'mobile': 'Mobile Experience',
            'comparison': 'Price Comparison'
        }
        
        # Generate events for the journey
        previous_url = None
        converted = False
        total_booking_value = 0
        
        for i, event_type in enumerate(event_sequence):
            # Calculate event timestamp with realistic gaps
            if i == 0:
                event_timestamp = journey_start
            else:
                # Variable time gaps based on event category
                prev_category = event_details.get(event_sequence[i-1], {}).get('category', '')
                curr_category = event_details.get(event_type, {}).get('category', '')
                
                if prev_category == curr_category:
                    gap_seconds = random.randint(15, 120)  # 15 seconds to 2 minutes for related actions
                elif curr_category == 'booking':
                    gap_seconds = random.randint(45, 300)  # Longer for booking steps
                else:
                    gap_seconds = random.randint(30, 600)  # 30 seconds to 10 minutes for category changes
                
                event_timestamp = previous_timestamp + timedelta(seconds=gap_seconds)
            
            previous_timestamp = event_timestamp
            
            # Get event information
            event_info = event_details.get(event_type, {
                'name': event_type.replace('_', ' ').title(),
                'url': f'/{event_type.replace("_", "-")}',
                'type': 'general',
                'category': 'other'
            })
            
            # Hotel details for relevant events
            event_hotel_name = hotel_name if event_info['category'] in ['property', 'booking', 'confirmation'] else None
            event_hotel_brand = hotel_brand if event_hotel_name else None
            event_destination_city = destination_city if event_info['category'] in ['search', 'property', 'booking'] else None
            event_destination_country = destination_country if event_destination_city else None
            event_room_type = room_type if event_info['category'] == 'booking' else None
            event_rate_plan = rate_plan if event_info['category'] == 'booking' else None
            event_room_rate = room_rate if event_info['category'] in ['property', 'booking'] else None
            
            # Booking details for booking events
            event_check_in = check_in_date if event_info['category'] in ['search', 'booking'] else None
            event_check_out = check_out_date if event_check_in else None
            event_nights = nights_stay if event_check_in else None
            event_guests = guests_count if event_check_in else None
            
            # Currency (simplified to USD for this example)
            currency_code = 'USD'
            
            # Booking value and revenue calculation
            total_booking_value = None
            revenue_impact = None
            
            # Determine if this is a conversion event
            conversion_events = ['booking_confirmation', 'confirmation_email_sent']
            is_conversion_event = event_type in conversion_events
            
            if is_conversion_event and random.random() < journey_template['conversion_rate']:
                converted = True
                if journey_template['revenue_range'][1] > 0:
                    total_booking_value = round(room_rate * nights_stay * random.uniform(0.8, 1.2), 2)
                    revenue_impact = total_booking_value
                elif journey_template['revenue_range'][0] < 0:  # Cancellation/modification
                    revenue_impact = journey_template['revenue_range'][0]
            
            # Custom events with explicit names (counts)
            hotel_searches = 1 if event_info['category'] == 'search' else 0
            property_views = 1 if event_info['category'] == 'property' else 0
            booking_starts = 1 if event_type in ['room_selection', 'guest_info_entry'] else 0
            booking_completions = 1 if event_type == 'booking_confirmation' else 0
            cancellation_requests = 1 if event_type == 'cancellation_request' else 0
            
            # Page interaction metrics
            time_on_page = random.randint(10, 900)  # 10 seconds to 15 minutes
            scroll_depth = random.randint(15, 100)  # Percentage
            clicks_on_page = random.randint(0, 30)
            page_load_time = random.randint(200, 5000)  # milliseconds
            
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
                
                # Hotel/Travel specific fields
                event_hotel_name,  # hotel_name
                event_hotel_brand,  # hotel_brand
                event_destination_city,  # destination_city
                event_destination_country,  # destination_country
                event_room_type,  # room_type
                event_rate_plan,  # rate_plan
                event_check_in,  # check_in_date
                event_check_out,  # check_out_date
                event_nights,  # nights_stay
                event_guests,  # guests_count
                event_room_rate,  # room_rate
                total_booking_value,  # total_booking_value
                currency_code,  # currency_code
                
                # Campaign/Marketing (consistent for journey)
                campaign_id,  # campaign_id
                traffic_source,  # traffic_source
                medium,  # medium
                referrer_domain,  # referrer_domain
                
                # Custom dimensions with explicit names (consistent for journey)
                traveler_type,  # traveler_type
                booking_purpose,  # booking_purpose
                loyalty_tier,  # loyalty_tier
                advance_days,  # advance_booking_days
                price_sensitivity,  # price_sensitivity
                
                # Custom events with explicit names
                hotel_searches,  # hotel_searches
                property_views,  # property_views
                booking_starts,  # booking_starts
                booking_completions,  # booking_completions
                cancellation_requests,  # cancellation_requests
                
                # Additional context
                is_mobile_app,  # is_mobile_app
                page_load_time,  # page_load_time_ms
                is_conversion_event and converted,  # conversion_flag
                revenue_impact  # revenue_impact
            )
            
            # Set previous URL for next iteration
            previous_url = event_info['url']
$$;

CREATE OR REPLACE TABLE hospitality_event_stream AS
SELECT e.*
FROM TABLE(GENERATOR(ROWCOUNT => 1000000)) g
CROSS JOIN TABLE(generate_hotel_journey()) e;

select * from hospitality_event_stream;
