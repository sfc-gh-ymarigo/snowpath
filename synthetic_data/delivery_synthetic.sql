CREATE OR REPLACE FUNCTION generate_food_delivery_journey()
RETURNS TABLE (
    -- Core identifiers
    event_id STRING,
    user_id STRING,
    customer_id STRING,
    session_id STRING,
    order_id STRING,
    
    -- Event details
    event_timestamp TIMESTAMP,
    event_type STRING,
    event_category STRING,
    event_action STRING,
    event_label STRING,
    
    -- App/Platform information
    platform STRING,
    app_version STRING,
    device_model STRING,
    operating_system STRING,
    user_agent STRING,
    
    -- Geographic data
    country STRING,
    state STRING,
    city STRING,
    zip_code STRING,
    delivery_zone STRING,
    
    -- Restaurant and food data
    restaurant_name STRING,
    restaurant_category STRING,
    cuisine_type STRING,
    restaurant_rating DECIMAL(3,2),
    delivery_time_estimate INT,
    item_name STRING,
    item_category STRING,
    item_price DECIMAL(8,2),
    
    -- Order details
    order_subtotal DECIMAL(10,2),
    delivery_fee DECIMAL(6,2),
    service_fee DECIMAL(6,2),
    tip_amount DECIMAL(8,2),
    taxes DECIMAL(8,2),
    total_order_value DECIMAL(12,2),
    payment_method STRING,
    
    -- Delivery information
    delivery_address_type STRING,
    estimated_delivery_time INT,
    actual_delivery_time INT,
    delivery_instructions STRING,
    driver_rating DECIMAL(3,2),
    
    -- Marketing and engagement
    campaign_id STRING,
    promo_code_used STRING,
    discount_amount DECIMAL(8,2),
    notification_type STRING,
    email_campaign_name STRING,
    
    -- Customer behavior dimensions
    customer_segment STRING,
    order_frequency_tier STRING,
    spending_tier STRING,
    preferred_cuisine STRING,
    dietary_preferences STRING,
    
    -- Custom food delivery events
    restaurant_views INT,
    menu_item_views INT,
    cart_additions INT,
    order_placements INT,
    reorders INT,
    
    -- Additional context
    is_first_order BOOLEAN,
    is_peak_hours BOOLEAN,
    weather_condition STRING,
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
        # Define shared event pools for food delivery experiences
        shared_events = {
            'app_entry': [
                'app_launch', 'push_notification_click', 'email_campaign_click', 'sms_link_click',
                'deeplink_open', 'widget_interaction', 'home_screen_shortcut', 'voice_assistant_open'
            ],
            'authentication': [
                'login_attempt', 'login_success', 'guest_order_start', 'account_creation_start',
                'social_login', 'phone_verification', 'email_verification', 'biometric_login'
            ],
            'location_services': [
                'location_permission_request', 'location_detected', 'address_entry', 'address_selection',
                'address_validation', 'delivery_zone_check', 'location_update', 'favorite_address_select'
            ],
            'restaurant_discovery': [
                'restaurant_feed_view', 'category_browse', 'cuisine_filter', 'rating_filter',
                'distance_filter', 'delivery_time_filter', 'price_range_filter', 'search_restaurants',
                'featured_restaurants_view', 'nearby_restaurants_view', 'trending_restaurants_view'
            ],
            'restaurant_interaction': [
                'restaurant_profile_view', 'menu_browse', 'item_details_view', 'photo_gallery_view',
                'reviews_section_view', 'restaurant_info_view', 'hours_check', 'contact_info_view',
                'favorite_restaurant_add', 'share_restaurant'
            ],
            'menu_navigation': [
                'menu_category_select', 'item_search', 'item_filter_apply', 'popular_items_view',
                'recommended_items_view', 'combo_deals_view', 'add_ons_view', 'customization_options',
                'nutritional_info_view', 'allergen_info_view'
            ],
            'cart_management': [
                'add_to_cart', 'cart_view', 'quantity_update', 'item_remove', 'item_customize',
                'special_instructions_add', 'cart_save_later', 'cart_share', 'similar_items_view',
                'upsell_item_view'
            ],
            'checkout_process': [
                'checkout_initiate', 'delivery_time_select', 'payment_method_select', 'tip_amount_select',
                'promo_code_apply', 'order_review', 'order_confirmation', 'payment_processing',
                'order_placed_success', 'receipt_view'
            ],
            'order_tracking': [
                'order_status_check', 'restaurant_preparing', 'driver_assigned', 'driver_pickup',
                'delivery_in_progress', 'delivery_eta_update', 'driver_location_track', 'delivery_arrived',
                'order_delivered', 'delivery_photo_view'
            ],
            'rating_feedback': [
                'restaurant_rating', 'driver_rating', 'order_rating', 'review_submission',
                'photo_review_upload', 'feedback_survey', 'complaint_submission', 'compliment_submission'
            ],
            'loyalty_rewards': [
                'loyalty_points_check', 'rewards_catalog_view', 'points_redemption', 'tier_status_check',
                'cashback_view', 'referral_program_use', 'milestone_achievement', 'bonus_points_earned'
            ],
            'customer_service': [
                'help_center_view', 'faq_browse', 'live_chat_start', 'call_support_request',
                'order_issue_report', 'refund_request', 'driver_feedback', 'restaurant_complaint',
                'missing_items_report', 'delivery_delay_report'
            ],
            'marketing_engagement': [
                'push_notification_receive', 'email_open', 'sms_receive', 'in_app_banner_click',
                'flash_sale_view', 'daily_deal_view', 'personalized_offer_view', 'group_order_invite',
                'social_share_deal', 'newsletter_signup'
            ],
            'social_features': [
                'group_order_create', 'group_order_join', 'friends_orders_view', 'social_feed_view',
                'restaurant_recommendation_send', 'order_history_share', 'wishlist_create',
                'follow_friends', 'create_food_list', 'join_food_challenge'
            ],
            'account_management': [
                'profile_update', 'payment_methods_manage', 'addresses_manage', 'preferences_update',
                'order_history_view', 'favorite_restaurants_view', 'dietary_preferences_set',
                'notification_settings', 'privacy_settings', 'subscription_manage'
            ],
            'session_end': [
                'app_minimize', 'logout', 'session_timeout', 'app_crash', 'network_disconnect',
                'background_mode', 'force_close', 'natural_exit'
            ]
        }
        
        # Define journey templates for food delivery experiences
        journey_templates = {
            'first_time_user_order': {
                'primary_goal': 'first_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', random.randint(1, 2)),
                    ('location_services', random.randint(1, 3)),
                    ('restaurant_discovery', random.randint(3, 6)),
                    ('restaurant_interaction', random.randint(2, 4)),
                    ('menu_navigation', random.randint(2, 5)),
                    ('cart_management', random.randint(2, 4)),
                    ('checkout_process', random.randint(4, 8)),
                    ('order_tracking', random.randint(2, 4)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.25,
                'revenue_range': (15, 45)
            },
            'regular_customer_reorder': {
                'primary_goal': 'repeat_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', random.randint(0, 1)),
                    ('restaurant_discovery', random.randint(1, 2)),
                    ('restaurant_interaction', random.randint(1, 2)),
                    ('menu_navigation', random.randint(1, 3)),
                    ('cart_management', random.randint(1, 2)),
                    ('checkout_process', random.randint(3, 5)),
                    ('order_tracking', random.randint(2, 4)),
                    ('rating_feedback', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.65,
                'revenue_range': (20, 60)
            },
            'deal_hunting_session': {
                'primary_goal': 'deal_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('marketing_engagement', random.randint(1, 3)),
                    ('restaurant_discovery', random.randint(2, 4)),
                    ('restaurant_interaction', random.randint(2, 4)),
                    ('menu_navigation', random.randint(1, 3)),
                    ('cart_management', random.randint(1, 3)),
                    ('checkout_process', random.randint(3, 6)),
                    ('order_tracking', random.randint(1, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.45,
                'revenue_range': (12, 35)
            },
            'browsing_no_order': {
                'primary_goal': 'exploration',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', random.randint(0, 1)),
                    ('location_services', random.randint(0, 1)),
                    ('restaurant_discovery', random.randint(3, 7)),
                    ('restaurant_interaction', random.randint(2, 5)),
                    ('menu_navigation', random.randint(1, 4)),
                    ('cart_management', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.05,
                'revenue_range': (0, 0)
            },
            'group_order_coordination': {
                'primary_goal': 'group_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', 1),
                    ('social_features', random.randint(2, 4)),
                    ('restaurant_discovery', random.randint(1, 3)),
                    ('restaurant_interaction', random.randint(1, 2)),
                    ('menu_navigation', random.randint(2, 4)),
                    ('cart_management', random.randint(2, 4)),
                    ('checkout_process', random.randint(4, 7)),
                    ('order_tracking', random.randint(2, 4)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.55,
                'revenue_range': (40, 120)
            },
            'premium_dining_experience': {
                'primary_goal': 'premium_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', 1),
                    ('restaurant_discovery', random.randint(2, 4)),
                    ('restaurant_interaction', random.randint(3, 6)),
                    ('menu_navigation', random.randint(2, 5)),
                    ('cart_management', random.randint(2, 4)),
                    ('checkout_process', random.randint(4, 6)),
                    ('order_tracking', random.randint(3, 5)),
                    ('rating_feedback', random.randint(1, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.70,
                'revenue_range': (35, 100)
            },
            'loyalty_member_session': {
                'primary_goal': 'loyalty_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', 1),
                    ('loyalty_rewards', random.randint(1, 3)),
                    ('restaurant_discovery', random.randint(1, 3)),
                    ('restaurant_interaction', random.randint(1, 3)),
                    ('menu_navigation', random.randint(1, 3)),
                    ('cart_management', random.randint(1, 2)),
                    ('checkout_process', random.randint(3, 5)),
                    ('order_tracking', random.randint(2, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.75,
                'revenue_range': (18, 55)
            },
            'customer_service_interaction': {
                'primary_goal': 'issue_resolution',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', random.randint(0, 1)),
                    ('account_management', random.randint(1, 2)),
                    ('customer_service', random.randint(3, 6)),
                    ('order_tracking', random.randint(0, 2)),
                    ('rating_feedback', random.randint(0, 1)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.30,
                'revenue_range': (0, 25)
            },
            'marketing_response_order': {
                'primary_goal': 'campaign_conversion',
                'base_flow': [
                    ('marketing_engagement', 1),
                    ('app_entry', 1),
                    ('authentication', random.randint(0, 1)),
                    ('restaurant_discovery', random.randint(1, 2)),
                    ('restaurant_interaction', random.randint(1, 3)),
                    ('menu_navigation', random.randint(1, 3)),
                    ('cart_management', random.randint(1, 3)),
                    ('checkout_process', random.randint(3, 6)),
                    ('order_tracking', random.randint(1, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.40,
                'revenue_range': (16, 48)
            },
            'late_night_craving': {
                'primary_goal': 'late_night_order',
                'base_flow': [
                    ('app_entry', 1),
                    ('authentication', random.randint(0, 1)),
                    ('restaurant_discovery', random.randint(1, 3)),
                    ('restaurant_interaction', random.randint(1, 2)),
                    ('menu_navigation', random.randint(1, 2)),
                    ('cart_management', random.randint(1, 2)),
                    ('checkout_process', random.randint(3, 5)),
                    ('order_tracking', random.randint(2, 4)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.50,
                'revenue_range': (8, 30)
            }
        }
        
        # Detailed event mappings for food delivery
        event_details = {
            # App Entry
            'app_launch': {'name': 'App Launch', 'url': '/app/launch', 'type': 'system', 'category': 'app_entry'},
            'push_notification_click': {'name': 'Push Notification Click', 'url': '/notification/click', 'type': 'marketing', 'category': 'app_entry'},
            'email_campaign_click': {'name': 'Email Campaign Click', 'url': '/email/click', 'type': 'marketing', 'category': 'app_entry'},
            'sms_link_click': {'name': 'SMS Link Click', 'url': '/sms/click', 'type': 'marketing', 'category': 'app_entry'},
            'deeplink_open': {'name': 'Deeplink Open', 'url': '/deeplink', 'type': 'system', 'category': 'app_entry'},
            'widget_interaction': {'name': 'Widget Interaction', 'url': '/widget', 'type': 'system', 'category': 'app_entry'},
            'home_screen_shortcut': {'name': 'Home Screen Shortcut', 'url': '/shortcut', 'type': 'system', 'category': 'app_entry'},
            'voice_assistant_open': {'name': 'Voice Assistant Open', 'url': '/voice', 'type': 'system', 'category': 'app_entry'},
            
            # Authentication
            'login_attempt': {'name': 'Login Attempt', 'url': '/auth/login', 'type': 'authentication', 'category': 'auth'},
            'login_success': {'name': 'Login Success', 'url': '/auth/success', 'type': 'authentication', 'category': 'auth'},
            'guest_order_start': {'name': 'Guest Order Start', 'url': '/auth/guest', 'type': 'authentication', 'category': 'auth'},
            'account_creation_start': {'name': 'Account Creation', 'url': '/auth/register', 'type': 'authentication', 'category': 'auth'},
            'social_login': {'name': 'Social Login', 'url': '/auth/social', 'type': 'authentication', 'category': 'auth'},
            'phone_verification': {'name': 'Phone Verification', 'url': '/auth/phone', 'type': 'authentication', 'category': 'auth'},
            'email_verification': {'name': 'Email Verification', 'url': '/auth/email', 'type': 'authentication', 'category': 'auth'},
            'biometric_login': {'name': 'Biometric Login', 'url': '/auth/biometric', 'type': 'authentication', 'category': 'auth'},
            
            # Location Services
            'location_permission_request': {'name': 'Location Permission', 'url': '/location/permission', 'type': 'system', 'category': 'location'},
            'location_detected': {'name': 'Location Detected', 'url': '/location/detected', 'type': 'system', 'category': 'location'},
            'address_entry': {'name': 'Address Entry', 'url': '/location/address', 'type': 'input', 'category': 'location'},
            'address_selection': {'name': 'Address Selection', 'url': '/location/select', 'type': 'selection', 'category': 'location'},
            'address_validation': {'name': 'Address Validation', 'url': '/location/validate', 'type': 'system', 'category': 'location'},
            'delivery_zone_check': {'name': 'Delivery Zone Check', 'url': '/location/zone', 'type': 'system', 'category': 'location'},
            'location_update': {'name': 'Location Update', 'url': '/location/update', 'type': 'input', 'category': 'location'},
            'favorite_address_select': {'name': 'Favorite Address Select', 'url': '/location/favorite', 'type': 'selection', 'category': 'location'},
            
            # Restaurant Discovery
            'restaurant_feed_view': {'name': 'Restaurant Feed', 'url': '/restaurants/feed', 'type': 'browse', 'category': 'discovery'},
            'category_browse': {'name': 'Category Browse', 'url': '/restaurants/category', 'type': 'browse', 'category': 'discovery'},
            'cuisine_filter': {'name': 'Cuisine Filter', 'url': '/restaurants/filter/cuisine', 'type': 'filter', 'category': 'discovery'},
            'rating_filter': {'name': 'Rating Filter', 'url': '/restaurants/filter/rating', 'type': 'filter', 'category': 'discovery'},
            'distance_filter': {'name': 'Distance Filter', 'url': '/restaurants/filter/distance', 'type': 'filter', 'category': 'discovery'},
            'delivery_time_filter': {'name': 'Delivery Time Filter', 'url': '/restaurants/filter/time', 'type': 'filter', 'category': 'discovery'},
            'price_range_filter': {'name': 'Price Range Filter', 'url': '/restaurants/filter/price', 'type': 'filter', 'category': 'discovery'},
            'search_restaurants': {'name': 'Search Restaurants', 'url': '/restaurants/search', 'type': 'search', 'category': 'discovery'},
            'featured_restaurants_view': {'name': 'Featured Restaurants', 'url': '/restaurants/featured', 'type': 'browse', 'category': 'discovery'},
            'nearby_restaurants_view': {'name': 'Nearby Restaurants', 'url': '/restaurants/nearby', 'type': 'browse', 'category': 'discovery'},
            'trending_restaurants_view': {'name': 'Trending Restaurants', 'url': '/restaurants/trending', 'type': 'browse', 'category': 'discovery'},
            
            # Restaurant Interaction
            'restaurant_profile_view': {'name': 'Restaurant Profile', 'url': '/restaurant/profile', 'type': 'view', 'category': 'restaurant'},
            'menu_browse': {'name': 'Menu Browse', 'url': '/restaurant/menu', 'type': 'browse', 'category': 'restaurant'},
            'item_details_view': {'name': 'Item Details', 'url': '/restaurant/item', 'type': 'view', 'category': 'restaurant'},
            'photo_gallery_view': {'name': 'Photo Gallery', 'url': '/restaurant/photos', 'type': 'view', 'category': 'restaurant'},
            'reviews_section_view': {'name': 'Reviews Section', 'url': '/restaurant/reviews', 'type': 'view', 'category': 'restaurant'},
            'restaurant_info_view': {'name': 'Restaurant Info', 'url': '/restaurant/info', 'type': 'view', 'category': 'restaurant'},
            'hours_check': {'name': 'Hours Check', 'url': '/restaurant/hours', 'type': 'view', 'category': 'restaurant'},
            'contact_info_view': {'name': 'Contact Info', 'url': '/restaurant/contact', 'type': 'view', 'category': 'restaurant'},
            'favorite_restaurant_add': {'name': 'Add to Favorites', 'url': '/restaurant/favorite', 'type': 'action', 'category': 'restaurant'},
            'share_restaurant': {'name': 'Share Restaurant', 'url': '/restaurant/share', 'type': 'social', 'category': 'restaurant'},
            
            # Menu Navigation
            'menu_category_select': {'name': 'Menu Category Select', 'url': '/menu/category', 'type': 'navigation', 'category': 'menu'},
            'item_search': {'name': 'Item Search', 'url': '/menu/search', 'type': 'search', 'category': 'menu'},
            'item_filter_apply': {'name': 'Item Filter', 'url': '/menu/filter', 'type': 'filter', 'category': 'menu'},
            'popular_items_view': {'name': 'Popular Items', 'url': '/menu/popular', 'type': 'view', 'category': 'menu'},
            'recommended_items_view': {'name': 'Recommended Items', 'url': '/menu/recommended', 'type': 'view', 'category': 'menu'},
            'combo_deals_view': {'name': 'Combo Deals', 'url': '/menu/combos', 'type': 'view', 'category': 'menu'},
            'add_ons_view': {'name': 'Add-ons View', 'url': '/menu/addons', 'type': 'view', 'category': 'menu'},
            'customization_options': {'name': 'Customization Options', 'url': '/menu/customize', 'type': 'view', 'category': 'menu'},
            'nutritional_info_view': {'name': 'Nutritional Info', 'url': '/menu/nutrition', 'type': 'view', 'category': 'menu'},
            'allergen_info_view': {'name': 'Allergen Info', 'url': '/menu/allergens', 'type': 'view', 'category': 'menu'},
            
            # Cart Management
            'add_to_cart': {'name': 'Add to Cart', 'url': '/cart/add', 'type': 'action', 'category': 'cart'},
            'cart_view': {'name': 'Cart View', 'url': '/cart', 'type': 'view', 'category': 'cart'},
            'quantity_update': {'name': 'Quantity Update', 'url': '/cart/quantity', 'type': 'action', 'category': 'cart'},
            'item_remove': {'name': 'Item Remove', 'url': '/cart/remove', 'type': 'action', 'category': 'cart'},
            'item_customize': {'name': 'Item Customize', 'url': '/cart/customize', 'type': 'action', 'category': 'cart'},
            'special_instructions_add': {'name': 'Special Instructions', 'url': '/cart/instructions', 'type': 'input', 'category': 'cart'},
            'cart_save_later': {'name': 'Save Cart for Later', 'url': '/cart/save', 'type': 'action', 'category': 'cart'},
            'cart_share': {'name': 'Share Cart', 'url': '/cart/share', 'type': 'social', 'category': 'cart'},
            'similar_items_view': {'name': 'Similar Items', 'url': '/cart/similar', 'type': 'view', 'category': 'cart'},
            'upsell_item_view': {'name': 'Upsell Items', 'url': '/cart/upsell', 'type': 'view', 'category': 'cart'},
            
            # Checkout Process
            'checkout_initiate': {'name': 'Checkout Start', 'url': '/checkout', 'type': 'action', 'category': 'checkout'},
            'delivery_time_select': {'name': 'Delivery Time Select', 'url': '/checkout/time', 'type': 'selection', 'category': 'checkout'},
            'payment_method_select': {'name': 'Payment Method', 'url': '/checkout/payment', 'type': 'selection', 'category': 'checkout'},
            'tip_amount_select': {'name': 'Tip Amount Select', 'url': '/checkout/tip', 'type': 'selection', 'category': 'checkout'},
            'promo_code_apply': {'name': 'Promo Code Apply', 'url': '/checkout/promo', 'type': 'action', 'category': 'checkout'},
            'order_review': {'name': 'Order Review', 'url': '/checkout/review', 'type': 'view', 'category': 'checkout'},
            'order_confirmation': {'name': 'Order Confirmation', 'url': '/checkout/confirm', 'type': 'action', 'category': 'checkout'},
            'payment_processing': {'name': 'Payment Processing', 'url': '/checkout/process', 'type': 'system', 'category': 'checkout'},
            'order_placed_success': {'name': 'Order Placed', 'url': '/checkout/success', 'type': 'confirmation', 'category': 'checkout'},
            'receipt_view': {'name': 'Receipt View', 'url': '/checkout/receipt', 'type': 'view', 'category': 'checkout'},
            
            # Order Tracking
            'order_status_check': {'name': 'Order Status Check', 'url': '/order/status', 'type': 'view', 'category': 'tracking'},
            'restaurant_preparing': {'name': 'Restaurant Preparing', 'url': '/order/preparing', 'type': 'status', 'category': 'tracking'},
            'driver_assigned': {'name': 'Driver Assigned', 'url': '/order/driver', 'type': 'status', 'category': 'tracking'},
            'driver_pickup': {'name': 'Driver Pickup', 'url': '/order/pickup', 'type': 'status', 'category': 'tracking'},
            'delivery_in_progress': {'name': 'Delivery in Progress', 'url': '/order/delivery', 'type': 'status', 'category': 'tracking'},
            'delivery_eta_update': {'name': 'ETA Update', 'url': '/order/eta', 'type': 'status', 'category': 'tracking'},
            'driver_location_track': {'name': 'Driver Location', 'url': '/order/location', 'type': 'view', 'category': 'tracking'},
            'delivery_arrived': {'name': 'Delivery Arrived', 'url': '/order/arrived', 'type': 'status', 'category': 'tracking'},
            'order_delivered': {'name': 'Order Delivered', 'url': '/order/delivered', 'type': 'confirmation', 'category': 'tracking'},
            'delivery_photo_view': {'name': 'Delivery Photo', 'url': '/order/photo', 'type': 'view', 'category': 'tracking'},
            
            # Rating & Feedback
            'restaurant_rating': {'name': 'Restaurant Rating', 'url': '/feedback/restaurant', 'type': 'rating', 'category': 'feedback'},
            'driver_rating': {'name': 'Driver Rating', 'url': '/feedback/driver', 'type': 'rating', 'category': 'feedback'},
            'order_rating': {'name': 'Order Rating', 'url': '/feedback/order', 'type': 'rating', 'category': 'feedback'},
            'review_submission': {'name': 'Review Submission', 'url': '/feedback/review', 'type': 'input', 'category': 'feedback'},
            'photo_review_upload': {'name': 'Photo Review Upload', 'url': '/feedback/photo', 'type': 'upload', 'category': 'feedback'},
            'feedback_survey': {'name': 'Feedback Survey', 'url': '/feedback/survey', 'type': 'survey', 'category': 'feedback'},
            'complaint_submission': {'name': 'Complaint Submission', 'url': '/feedback/complaint', 'type': 'complaint', 'category': 'feedback'},
            'compliment_submission': {'name': 'Compliment Submission', 'url': '/feedback/compliment', 'type': 'compliment', 'category': 'feedback'},
            
            # Loyalty & Rewards
            'loyalty_points_check': {'name': 'Loyalty Points Check', 'url': '/loyalty/points', 'type': 'view', 'category': 'loyalty'},
            'rewards_catalog_view': {'name': 'Rewards Catalog', 'url': '/loyalty/catalog', 'type': 'view', 'category': 'loyalty'},
            'points_redemption': {'name': 'Points Redemption', 'url': '/loyalty/redeem', 'type': 'action', 'category': 'loyalty'},
            'tier_status_check': {'name': 'Tier Status Check', 'url': '/loyalty/tier', 'type': 'view', 'category': 'loyalty'},
            'cashback_view': {'name': 'Cashback View', 'url': '/loyalty/cashback', 'type': 'view', 'category': 'loyalty'},
            'referral_program_use': {'name': 'Referral Program', 'url': '/loyalty/referral', 'type': 'action', 'category': 'loyalty'},
            'milestone_achievement': {'name': 'Milestone Achievement', 'url': '/loyalty/milestone', 'type': 'achievement', 'category': 'loyalty'},
            'bonus_points_earned': {'name': 'Bonus Points Earned', 'url': '/loyalty/bonus', 'type': 'achievement', 'category': 'loyalty'},
            
            # Customer Service
            'help_center_view': {'name': 'Help Center', 'url': '/support/help', 'type': 'view', 'category': 'support'},
            'faq_browse': {'name': 'FAQ Browse', 'url': '/support/faq', 'type': 'browse', 'category': 'support'},
            'live_chat_start': {'name': 'Live Chat Start', 'url': '/support/chat', 'type': 'action', 'category': 'support'},
            'call_support_request': {'name': 'Call Support Request', 'url': '/support/call', 'type': 'action', 'category': 'support'},
            'order_issue_report': {'name': 'Order Issue Report', 'url': '/support/issue', 'type': 'report', 'category': 'support'},
            'refund_request': {'name': 'Refund Request', 'url': '/support/refund', 'type': 'request', 'category': 'support'},
            'driver_feedback': {'name': 'Driver Feedback', 'url': '/support/driver', 'type': 'feedback', 'category': 'support'},
            'restaurant_complaint': {'name': 'Restaurant Complaint', 'url': '/support/restaurant', 'type': 'complaint', 'category': 'support'},
            'missing_items_report': {'name': 'Missing Items Report', 'url': '/support/missing', 'type': 'report', 'category': 'support'},
            'delivery_delay_report': {'name': 'Delivery Delay Report', 'url': '/support/delay', 'type': 'report', 'category': 'support'},
            
            # Marketing Engagement
            'push_notification_receive': {'name': 'Push Notification Receive', 'url': '/marketing/push', 'type': 'receive', 'category': 'marketing'},
            'email_open': {'name': 'Email Open', 'url': '/marketing/email', 'type': 'open', 'category': 'marketing'},
            'sms_receive': {'name': 'SMS Receive', 'url': '/marketing/sms', 'type': 'receive', 'category': 'marketing'},
            'in_app_banner_click': {'name': 'In-App Banner Click', 'url': '/marketing/banner', 'type': 'click', 'category': 'marketing'},
            'flash_sale_view': {'name': 'Flash Sale View', 'url': '/marketing/flash', 'type': 'view', 'category': 'marketing'},
            'daily_deal_view': {'name': 'Daily Deal View', 'url': '/marketing/daily', 'type': 'view', 'category': 'marketing'},
            'personalized_offer_view': {'name': 'Personalized Offer', 'url': '/marketing/personalized', 'type': 'view', 'category': 'marketing'},
            'group_order_invite': {'name': 'Group Order Invite', 'url': '/marketing/group', 'type': 'invite', 'category': 'marketing'},
            'social_share_deal': {'name': 'Social Share Deal', 'url': '/marketing/share', 'type': 'share', 'category': 'marketing'},
            'newsletter_signup': {'name': 'Newsletter Signup', 'url': '/marketing/newsletter', 'type': 'signup', 'category': 'marketing'},
            
            # Social Features
            'group_order_create': {'name': 'Group Order Create', 'url': '/social/group/create', 'type': 'create', 'category': 'social'},
            'group_order_join': {'name': 'Group Order Join', 'url': '/social/group/join', 'type': 'join', 'category': 'social'},
            'friends_orders_view': {'name': 'Friends Orders View', 'url': '/social/friends', 'type': 'view', 'category': 'social'},
            'social_feed_view': {'name': 'Social Feed View', 'url': '/social/feed', 'type': 'view', 'category': 'social'},
            'restaurant_recommendation_send': {'name': 'Restaurant Recommendation', 'url': '/social/recommend', 'type': 'share', 'category': 'social'},
            'order_history_share': {'name': 'Order History Share', 'url': '/social/history', 'type': 'share', 'category': 'social'},
            'wishlist_create': {'name': 'Wishlist Create', 'url': '/social/wishlist', 'type': 'create', 'category': 'social'},
            'follow_friends': {'name': 'Follow Friends', 'url': '/social/follow', 'type': 'follow', 'category': 'social'},
            'create_food_list': {'name': 'Create Food List', 'url': '/social/list', 'type': 'create', 'category': 'social'},
            'join_food_challenge': {'name': 'Join Food Challenge', 'url': '/social/challenge', 'type': 'join', 'category': 'social'},
            
            # Account Management
            'profile_update': {'name': 'Profile Update', 'url': '/account/profile', 'type': 'update', 'category': 'account'},
            'payment_methods_manage': {'name': 'Payment Methods', 'url': '/account/payment', 'type': 'manage', 'category': 'account'},
            'addresses_manage': {'name': 'Addresses Manage', 'url': '/account/addresses', 'type': 'manage', 'category': 'account'},
            'preferences_update': {'name': 'Preferences Update', 'url': '/account/preferences', 'type': 'update', 'category': 'account'},
            'order_history_view': {'name': 'Order History View', 'url': '/account/orders', 'type': 'view', 'category': 'account'},
            'favorite_restaurants_view': {'name': 'Favorite Restaurants', 'url': '/account/favorites', 'type': 'view', 'category': 'account'},
            'dietary_preferences_set': {'name': 'Dietary Preferences', 'url': '/account/dietary', 'type': 'set', 'category': 'account'},
            'notification_settings': {'name': 'Notification Settings', 'url': '/account/notifications', 'type': 'settings', 'category': 'account'},
            'privacy_settings': {'name': 'Privacy Settings', 'url': '/account/privacy', 'type': 'settings', 'category': 'account'},
            'subscription_manage': {'name': 'Subscription Manage', 'url': '/account/subscription', 'type': 'manage', 'category': 'account'},
            
            # Session End
            'app_minimize': {'name': 'App Minimize', 'url': '/app/minimize', 'type': 'system', 'category': 'session_end'},
            'logout': {'name': 'Logout', 'url': '/auth/logout', 'type': 'authentication', 'category': 'session_end'},
            'session_timeout': {'name': 'Session Timeout', 'url': '/session/timeout', 'type': 'system', 'category': 'session_end'},
            'app_crash': {'name': 'App Crash', 'url': '/app/crash', 'type': 'system', 'category': 'session_end'},
            'network_disconnect': {'name': 'Network Disconnect', 'url': '/network/disconnect', 'type': 'system', 'category': 'session_end'},
            'background_mode': {'name': 'Background Mode', 'url': '/app/background', 'type': 'system', 'category': 'session_end'},
            'force_close': {'name': 'Force Close', 'url': '/app/force_close', 'type': 'system', 'category': 'session_end'},
            'natural_exit': {'name': 'Natural Exit', 'url': '/app/exit', 'type': 'system', 'category': 'session_end'}
        }
        
        # Food delivery data
        restaurant_names = [
            'Pizza Palace', 'Burger Kingdom', 'Taco Fiesta', 'Sushi Zen', 'Noodle House',
            'BBQ Pit Master', 'Green Garden Salads', 'Spice Route Indian', 'Pasta La Vista',
            'Wings & Things', 'Thai Orchid', 'Mediterranean Grill', 'Sandwich Station',
            'Ice Cream Dreams', 'Coffee Corner', 'Breakfast Bistro', 'Seafood Shack',
            'Steakhouse Supreme', 'Vegan Vibes', 'Donut Delight', 'Smoothie Central',
            'Fried Chicken Express', 'Ramen Station', 'Greek Gyros', 'Mexican Cantina'
        ]
        
        restaurant_categories = [
            'Fast Food', 'Fast Casual', 'Casual Dining', 'Fine Dining', 'Coffee & Tea',
            'Desserts', 'Healthy', 'Comfort Food', 'Street Food', 'Bakery'
        ]
        
        cuisine_types = [
            'American', 'Italian', 'Mexican', 'Chinese', 'Japanese', 'Indian', 'Thai',
            'Mediterranean', 'Greek', 'Korean', 'Vietnamese', 'French', 'Spanish',
            'BBQ', 'Seafood', 'Vegetarian', 'Vegan', 'Halal', 'Kosher'
        ]
        
        item_categories = [
            'appetizers', 'entrees', 'sides', 'desserts', 'beverages', 'salads',
            'soups', 'sandwiches', 'pizza', 'pasta', 'burgers', 'tacos'
        ]
        
        payment_methods = [
            'credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay',
            'venmo', 'cash', 'gift_card', 'loyalty_points', 'corporate_card'
        ]
        
        delivery_address_types = ['home', 'work', 'hotel', 'friend', 'other']
        
        customer_segments = [
            'new_user', 'occasional_orderer', 'regular_customer', 'frequent_orderer',
            'premium_customer', 'bargain_hunter', 'food_explorer', 'convenience_seeker'
        ]
        
        order_frequency_tiers = ['first_time', 'occasional', 'regular', 'frequent', 'daily']
        spending_tiers = ['budget', 'moderate', 'premium', 'high_value']
        
        dietary_preferences = [
            'none', 'vegetarian', 'vegan', 'gluten_free', 'keto', 'paleo',
            'dairy_free', 'nut_allergy', 'low_sodium', 'diabetic_friendly'
        ]
        
        platforms = ['iOS', 'Android', 'Web']
        device_models = [
            'iPhone 15', 'iPhone 14', 'iPhone 13', 'Samsung Galaxy S24', 'Samsung Galaxy S23',
            'Google Pixel 8', 'OnePlus 12', 'iPad Pro', 'Samsung Tablet', 'Desktop'
        ]
        operating_systems = ['iOS 17', 'iOS 16', 'Android 14', 'Android 13', 'Windows 11', 'macOS 14']
        
        notification_types = ['push', 'email', 'sms', 'in_app']
        weather_conditions = ['sunny', 'rainy', 'snowy', 'cloudy', 'stormy', 'hot', 'cold']
        
        # Generate consistent user profile for this journey
        user_id = str(uuid.uuid4())
        customer_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        order_id = str(uuid.uuid4()) if random.random() < 0.7 else None  # 70% have order_id
        
        customer_segment = random.choice(customer_segments)
        order_frequency_tier = random.choice(order_frequency_tiers)
        spending_tier = random.choice(spending_tiers)
        preferred_cuisine = random.choice(cuisine_types)
        dietary_preference = random.choice(dietary_preferences)
        
        # Technical details
        platform = random.choice(platforms)
        device_model = random.choice(device_models)
        os = random.choice(operating_systems)
        app_version = f"{random.randint(8, 12)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        user_agent = f"FoodDeliveryApp/{app_version} ({platform}; {os})"
        
        # Geographic data
        country = 'United States'
        state = fake.state()
        city = fake.city()
        zip_code = fake.zipcode()
        delivery_zone = f"Zone_{random.randint(1, 15)}"
        
        # Restaurant and food details
        restaurant_name = random.choice(restaurant_names)
        restaurant_category = random.choice(restaurant_categories)
        cuisine_type = random.choice(cuisine_types)
        restaurant_rating = round(random.uniform(3.0, 5.0), 2)
        delivery_time_estimate = random.randint(15, 60)
        
        # Order details
        payment_method = random.choice(payment_methods)
        delivery_address_type = random.choice(delivery_address_types)
        
        # Contextual data
        is_first_order = order_frequency_tier == 'first_time'
        hour = random.randint(0, 23)
        is_peak_hours = hour in [11, 12, 13, 17, 18, 19, 20]  # Lunch and dinner peaks
        weather_condition = random.choice(weather_conditions)
        
        # Choose a journey template
        journey_name = random.choice(list(journey_templates.keys()))
        journey_template = journey_templates[journey_name]
        
        # Build the actual event sequence from the template
        event_sequence = []
        for event_category, count in journey_template['base_flow']:
            selected_events = random.sample(shared_events[event_category], min(count, len(shared_events[event_category])))
            event_sequence.extend(selected_events)
        
        # Add some randomization - 20% chance to add extra events
        if random.random() < 0.20:
            extra_categories = [cat for cat in shared_events.keys() if cat not in ['session_end']]
            extra_category = random.choice(extra_categories)
            extra_event = random.choice(shared_events[extra_category])
            insert_pos = random.randint(1, len(event_sequence) - 1)
            event_sequence.insert(insert_pos, extra_event)
        
        # Generate journey start time
        journey_start = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Generate events for the journey
        converted = False
        total_order_value = 0
        
        for i, event_type in enumerate(event_sequence):
            # Calculate event timestamp
            if i == 0:
                event_timestamp = journey_start
            else:
                # Food delivery events have varied timing
                if event_details.get(event_type, {}).get('category') == 'tracking':
                    gap_seconds = random.randint(300, 1800)  # 5-30 minutes for tracking events
                elif event_details.get(event_type, {}).get('category') == 'marketing':
                    gap_seconds = random.randint(0, 60)  # Quick response to marketing
                else:
                    gap_seconds = random.randint(15, 300)  # 15 seconds to 5 minutes
                
                event_timestamp = previous_timestamp + timedelta(seconds=gap_seconds)
            
            previous_timestamp = event_timestamp
            
            # Get event information
            event_info = event_details.get(event_type, {
                'name': event_type.replace('_', ' ').title(),
                'url': f'/{event_type.replace("_", "/")}',
                'type': 'general',
                'category': 'other'
            })
            
            # Item and pricing details
            item_name = None
            item_category = None
            item_price = None
            
            if event_info['category'] in ['restaurant', 'menu', 'cart']:
                item_names = [
                    'Margherita Pizza', 'Cheeseburger Deluxe', 'Chicken Tacos', 'California Roll',
                    'Pad Thai', 'BBQ Ribs', 'Caesar Salad', 'Butter Chicken', 'Carbonara Pasta',
                    'Buffalo Wings', 'Tom Yum Soup', 'Gyro Platter', 'Club Sandwich',
                    'Chocolate Cake', 'Iced Coffee', 'Pancake Stack', 'Fish & Chips'
                ]
                item_name = random.choice(item_names)
                item_category = random.choice(item_categories)
                item_price = round(random.uniform(8, 35), 2)
            
            # Order calculations
            order_subtotal = None
            delivery_fee = None
            service_fee = None
            tip_amount = None
            taxes = None
            total_order_value = None
            discount_amount = None
            promo_code_used = None
            estimated_delivery_time = None
            actual_delivery_time = None
            delivery_instructions = None
            driver_rating = None
            revenue_impact = None
            
            # Marketing fields
            campaign_id = None
            notification_type = None
            email_campaign_name = None
            
            if event_info['category'] == 'marketing':
                campaign_id = str(uuid.uuid4())
                notification_type = random.choice(notification_types)
                email_campaign_name = random.choice([
                    'Weekend Special', 'Lunch Deal Alert', 'New Restaurant Launch', 
                    'Flash Sale', 'Loyalty Rewards', 'Weather-Based Offer'
                ])
            
            # Order-related calculations
            if event_info['category'] == 'checkout':
                order_subtotal = round(random.uniform(15, 80), 2)
                delivery_fee = round(random.uniform(1.99, 5.99), 2)
                service_fee = round(order_subtotal * 0.15, 2)  # 15% service fee
                tip_amount = round(order_subtotal * random.uniform(0.15, 0.25), 2)
                taxes = round(order_subtotal * 0.08875, 2)  # ~8.875% tax
                
                if random.random() < 0.3:  # 30% use promo codes
                    promo_codes = ['SAVE20', 'FREEDELIV', 'NEWUSER15', 'FLASH30', 'WELCOME10']
                    promo_code_used = random.choice(promo_codes)
                    discount_amount = round(order_subtotal * random.uniform(0.10, 0.30), 2)
                
                total_order_value = order_subtotal + delivery_fee + service_fee + tip_amount + taxes
                if discount_amount:
                    total_order_value -= discount_amount
                
                total_order_value = round(total_order_value, 2)
            
            if event_info['category'] == 'tracking':
                estimated_delivery_time = random.randint(25, 60)
                actual_delivery_time = estimated_delivery_time + random.randint(-10, 20)
                delivery_instructions = random.choice([
                    'Leave at door', 'Ring doorbell', 'Call when arrived', 
                    'Meet at lobby', 'Contactless delivery', None
                ])
            
            if event_type == 'driver_rating':
                driver_rating = round(random.uniform(3.0, 5.0), 2)
            
            # Conversion determination
            conversion_events = ['order_placed_success', 'order_delivered']
            is_conversion_event = event_type in conversion_events
            
            if is_conversion_event and random.random() < journey_template['conversion_rate']:
                converted = True
                if journey_template['revenue_range'][1] > 0:
                    revenue_impact = round(random.uniform(*journey_template['revenue_range']), 2)
            
            # Custom food delivery events (counts)
            restaurant_views = 1 if event_info['category'] in ['discovery', 'restaurant'] else 0
            menu_item_views = 1 if event_info['category'] == 'menu' else 0
            cart_additions = 1 if event_type == 'add_to_cart' else 0
            order_placements = 1 if event_type == 'order_placed_success' else 0
            reorders = 1 if event_type == 'add_to_cart' and order_frequency_tier != 'first_time' else 0
            
            yield (
                # Core identifiers
                str(uuid.uuid4()),  # event_id
                user_id,  # user_id (consistent)
                customer_id,  # customer_id (consistent)
                session_id,  # session_id (consistent)
                order_id,  # order_id
                
                # Event details
                event_timestamp,  # event_timestamp
                event_type,  # event_type
                event_info['category'],  # event_category
                event_type.replace('_', ' ').title(),  # event_action
                f"{journey_template['primary_goal']}_{event_type}",  # event_label
                
                # App/Platform information
                platform,  # platform
                app_version,  # app_version
                device_model,  # device_model
                os,  # operating_system
                user_agent,  # user_agent
                
                # Geographic data
                country,  # country
                state,  # state
                city,  # city
                zip_code,  # zip_code
                delivery_zone,  # delivery_zone
                
                # Restaurant and food data
                restaurant_name if event_info['category'] in ['restaurant', 'menu', 'cart', 'checkout'] else None,  # restaurant_name
                restaurant_category if event_info['category'] in ['restaurant', 'menu', 'cart', 'checkout'] else None,  # restaurant_category
                cuisine_type if event_info['category'] in ['restaurant', 'menu', 'cart', 'checkout'] else None,  # cuisine_type
                restaurant_rating if event_info['category'] in ['restaurant', 'feedback'] else None,  # restaurant_rating
                delivery_time_estimate if event_info['category'] in ['restaurant', 'checkout'] else None,  # delivery_time_estimate
                item_name,  # item_name
                item_category,  # item_category
                item_price,  # item_price
                
                # Order details
                order_subtotal,  # order_subtotal
                delivery_fee,  # delivery_fee
                service_fee,  # service_fee
                tip_amount,  # tip_amount
                taxes,  # taxes
                total_order_value,  # total_order_value
                payment_method if event_info['category'] == 'checkout' else None,  # payment_method
                
                # Delivery information
                delivery_address_type if event_info['category'] in ['checkout', 'tracking'] else None,  # delivery_address_type
                estimated_delivery_time,  # estimated_delivery_time
                actual_delivery_time,  # actual_delivery_time
                delivery_instructions,  # delivery_instructions
                driver_rating,  # driver_rating
                
                # Marketing and engagement
                campaign_id,  # campaign_id
                promo_code_used,  # promo_code_used
                discount_amount,  # discount_amount
                notification_type,  # notification_type
                email_campaign_name,  # email_campaign_name
                
                # Customer behavior dimensions
                customer_segment,  # customer_segment
                order_frequency_tier,  # order_frequency_tier
                spending_tier,  # spending_tier
                preferred_cuisine,  # preferred_cuisine
                dietary_preference,  # dietary_preferences
                
                # Custom food delivery events
                restaurant_views,  # restaurant_views
                menu_item_views,  # menu_item_views
                cart_additions,  # cart_additions
                order_placements,  # order_placements
                reorders,  # reorders
                
                # Additional context
                is_first_order,  # is_first_order
                is_peak_hours,  # is_peak_hours
                weather_condition,  # weather_condition
                is_conversion_event and converted,  # conversion_flag
                revenue_impact  # revenue_impact
            )
$$;

CREATE OR REPLACE TABLE delivery_app_event_stream AS
SELECT e.*
FROM TABLE(GENERATOR(ROWCOUNT => 1000000)) g
CROSS JOIN TABLE(generate_food_delivery_journey()) e;

select * from delivery_app_event_stream;
