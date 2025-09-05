CREATE OR REPLACE FUNCTION generate_ecommerce_journey()
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
    
    -- Ecommerce specific fields
    product_name STRING,
    product_category STRING,
    product_brand STRING,
    product_price DECIMAL(10,2),
    order_value DECIMAL(12,2),
    quantity INT,
    discount_amount DECIMAL(10,2),
    payment_method STRING,
    shipping_method STRING,
    
    -- Campaign/Marketing (Adobe Analytics style)
    campaign_id STRING,
    traffic_source STRING,
    medium STRING,
    referrer_domain STRING,
    
    -- Custom dimensions with explicit names
    customer_segment STRING,
    customer_lifetime_value_tier STRING,
    sport_preference STRING,
    size_preference STRING,
    loyalty_program_member STRING,
    
    -- Custom events with explicit names
    product_views INT,
    add_to_cart_events INT,
    purchase_events INT,
    search_events INT,
    wishlist_additions INT,
    
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
        # Define shared event pools for ecommerce sporting goods
        shared_events = {
            'entry_points': [
                'homepage_visit', 'category_landing', 'search_result_click', 'email_campaign_click',
                'social_media_click', 'mobile_app_open', 'influencer_link_click', 'google_ads_click',
                'affiliate_referral', 'direct_url_entry'
            ],
            'authentication': [
                'login_attempt', 'login_success', 'guest_checkout_start', 'account_creation_start',
                'password_reset_request', 'social_login_attempt', 'email_verification'
            ],
            'product_discovery': [
                'category_browse', 'product_listing_view', 'filter_application', 'sort_selection',
                'search_query', 'search_refinement', 'brand_page_visit', 'sale_section_browse',
                'new_arrivals_view', 'trending_products_view', 'size_guide_view'
            ],
            'product_interaction': [
                'product_detail_view', 'product_image_zoom', 'size_selection', 'color_selection',
                'product_video_play', 'review_section_view', 'qa_section_view', 'size_chart_view',
                'product_comparison', 'related_products_view', 'recently_viewed_check'
            ],
            'cart_interactions': [
                'add_to_cart', 'cart_view', 'quantity_update', 'remove_from_cart',
                'save_for_later', 'cart_abandonment_recovery', 'promo_code_entry',
                'shipping_calculator_use', 'cart_share'
            ],
            'checkout_process': [
                'checkout_initiation', 'shipping_info_entry', 'billing_info_entry',
                'payment_method_selection', 'order_review', 'purchase_completion',
                'order_confirmation_view', 'receipt_email_open'
            ],
            'account_management': [
                'account_dashboard_view', 'order_history_view', 'profile_update',
                'address_book_management', 'payment_methods_management', 'preferences_update',
                'subscription_management', 'loyalty_points_check'
            ],
            'wishlist_favorites': [
                'wishlist_view', 'add_to_wishlist', 'remove_from_wishlist',
                'wishlist_share', 'move_to_cart_from_wishlist', 'favorites_organization'
            ],
            'reviews_social': [
                'review_submission', 'review_reading', 'rating_submission',
                'photo_review_upload', 'social_share', 'referral_program_use',
                'user_generated_content_view'
            ],
            'support_touchpoints': [
                'help_center_visit', 'faq_browse', 'live_chat_initiate', 'contact_form_submission',
                'return_policy_view', 'shipping_info_view', 'size_exchange_request',
                'order_tracking', 'customer_service_call', 'chatbot_interaction'
            ],
            'mobile_specific': [
                'app_notification_click', 'barcode_scan', 'store_locator_use',
                'mobile_exclusive_offer_view', 'push_notification_settings',
                'mobile_payment_setup', 'offline_wishlist_sync'
            ],
            'promotional': [
                'coupon_code_search', 'sale_banner_click', 'loyalty_program_join',
                'email_signup', 'sms_signup', 'flash_sale_participation',
                'seasonal_promotion_view', 'bundle_offer_view'
            ],
            'cross_selling': [
                'recommended_products_view', 'frequently_bought_together',
                'personalized_recommendations', 'category_upsell_view',
                'accessory_suggestions', 'outfit_completion_suggestions'
            ],
            'exits': [
                'logout', 'session_timeout', 'navigation_away', 'app_background',
                'browser_close', 'checkout_abandonment'
            ]
        }
        
        # Define journey templates for sporting goods ecommerce
        journey_templates = {
            'new_customer_exploration': {
                'primary_goal': 'first_purchase',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_discovery', random.randint(3, 6)),
                    ('product_interaction', random.randint(2, 4)),
                    ('authentication', random.randint(1, 2)),
                    ('cart_interactions', random.randint(1, 3)),
                    ('checkout_process', random.randint(2, 6)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.15,
                'revenue_range': (25, 200)
            },
            'returning_customer_purchase': {
                'primary_goal': 'repeat_purchase',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('product_discovery', random.randint(1, 3)),
                    ('product_interaction', random.randint(1, 3)),
                    ('cart_interactions', random.randint(1, 2)),
                    ('checkout_process', random.randint(3, 5)),
                    ('account_management', random.randint(0, 1)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.35,
                'revenue_range': (40, 300)
            },
            'athletic_gear_research': {
                'primary_goal': 'product_research',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_discovery', random.randint(4, 7)),
                    ('product_interaction', random.randint(3, 6)),
                    ('reviews_social', random.randint(1, 3)),
                    ('support_touchpoints', random.randint(0, 2)),
                    ('wishlist_favorites', random.randint(0, 2)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.08,
                'revenue_range': (0, 0)
            },
            'seasonal_shopping_spree': {
                'primary_goal': 'bulk_purchase',
                'base_flow': [
                    ('entry_points', 1),
                    ('promotional', random.randint(1, 2)),
                    ('product_discovery', random.randint(3, 5)),
                    ('product_interaction', random.randint(4, 8)),
                    ('cart_interactions', random.randint(2, 4)),
                    ('cross_selling', random.randint(1, 3)),
                    ('checkout_process', random.randint(3, 6)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.45,
                'revenue_range': (150, 800)
            },
            'mobile_app_browsing': {
                'primary_goal': 'mobile_engagement',
                'base_flow': [
                    ('entry_points', 1),
                    ('mobile_specific', random.randint(2, 4)),
                    ('product_discovery', random.randint(2, 4)),
                    ('product_interaction', random.randint(1, 3)),
                    ('wishlist_favorites', random.randint(0, 2)),
                    ('cart_interactions', random.randint(0, 2)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.12,
                'revenue_range': (20, 150)
            },
            'gift_shopping_journey': {
                'primary_goal': 'gift_purchase',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_discovery', random.randint(2, 4)),
                    ('product_interaction', random.randint(2, 5)),
                    ('reviews_social', random.randint(1, 2)),
                    ('support_touchpoints', random.randint(0, 2)),
                    ('cart_interactions', random.randint(1, 3)),
                    ('checkout_process', random.randint(3, 6)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.28,
                'revenue_range': (30, 250)
            },
            'loyalty_member_shopping': {
                'primary_goal': 'loyalty_purchase',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', 1),
                    ('account_management', random.randint(1, 2)),
                    ('promotional', random.randint(1, 2)),
                    ('product_discovery', random.randint(2, 4)),
                    ('product_interaction', random.randint(1, 3)),
                    ('cart_interactions', random.randint(1, 2)),
                    ('checkout_process', random.randint(2, 4)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.55,
                'revenue_range': (60, 400)
            },
            'support_interaction': {
                'primary_goal': 'customer_service',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(0, 1)),
                    ('account_management', random.randint(1, 2)),
                    ('support_touchpoints', random.randint(3, 6)),
                    ('product_discovery', random.randint(0, 2)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.20,
                'revenue_range': (0, 100)
            },
            'cart_abandonment_recovery': {
                'primary_goal': 'abandoned_cart_return',
                'base_flow': [
                    ('entry_points', 1),
                    ('authentication', random.randint(0, 1)),
                    ('cart_interactions', random.randint(1, 3)),
                    ('promotional', random.randint(0, 1)),
                    ('checkout_process', random.randint(0, 4)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.25,
                'revenue_range': (25, 180)
            },
            'social_media_influenced': {
                'primary_goal': 'social_conversion',
                'base_flow': [
                    ('entry_points', 1),
                    ('product_interaction', random.randint(2, 4)),
                    ('reviews_social', random.randint(1, 3)),
                    ('wishlist_favorites', random.randint(0, 2)),
                    ('cart_interactions', random.randint(0, 2)),
                    ('checkout_process', random.randint(0, 5)),
                    ('exits', 1)
                ],
                'conversion_rate': 0.18,
                'revenue_range': (35, 220)
            }
        }
        
        # Detailed event mappings for sporting goods ecommerce
        event_details = {
            # Entry points
            'homepage_visit': {'name': 'Homepage', 'url': '/', 'type': 'marketing', 'category': 'navigation'},
            'category_landing': {'name': 'Category Landing', 'url': '/category/running-shoes', 'type': 'marketing', 'category': 'navigation'},
            'search_result_click': {'name': 'Search Results', 'url': '/search?q=nike', 'type': 'search', 'category': 'search'},
            'email_campaign_click': {'name': 'Email Campaign', 'url': '/campaign/new-arrivals', 'type': 'marketing', 'category': 'campaign'},
            'social_media_click': {'name': 'Social Media', 'url': '/social-landing', 'type': 'marketing', 'category': 'social'},
            'mobile_app_open': {'name': 'Mobile App Home', 'url': '/app/home', 'type': 'mobile', 'category': 'mobile'},
            'influencer_link_click': {'name': 'Influencer Link', 'url': '/influencer/athlete-gear', 'type': 'marketing', 'category': 'influencer'},
            'google_ads_click': {'name': 'Google Ads', 'url': '/ads-landing', 'type': 'marketing', 'category': 'paid_search'},
            'affiliate_referral': {'name': 'Affiliate Referral', 'url': '/affiliate/sports-blog', 'type': 'marketing', 'category': 'affiliate'},
            'direct_url_entry': {'name': 'Direct Entry', 'url': '/direct', 'type': 'direct', 'category': 'direct'},
            
            # Authentication
            'login_attempt': {'name': 'Login Page', 'url': '/login', 'type': 'authentication', 'category': 'auth'},
            'login_success': {'name': 'Login Success', 'url': '/my-account', 'type': 'authentication', 'category': 'auth'},
            'guest_checkout_start': {'name': 'Guest Checkout', 'url': '/checkout/guest', 'type': 'checkout', 'category': 'auth'},
            'account_creation_start': {'name': 'Create Account', 'url': '/register', 'type': 'authentication', 'category': 'auth'},
            'password_reset_request': {'name': 'Password Reset', 'url': '/forgot-password', 'type': 'authentication', 'category': 'auth'},
            'social_login_attempt': {'name': 'Social Login', 'url': '/login/social', 'type': 'authentication', 'category': 'auth'},
            'email_verification': {'name': 'Email Verification', 'url': '/verify-email', 'type': 'authentication', 'category': 'auth'},
            
            # Product Discovery
            'category_browse': {'name': 'Category Browse', 'url': '/category/athletic-wear', 'type': 'product_listing', 'category': 'browse'},
            'product_listing_view': {'name': 'Product Listing', 'url': '/products/running-shoes', 'type': 'product_listing', 'category': 'browse'},
            'filter_application': {'name': 'Apply Filters', 'url': '/products/shoes?filter=brand:nike', 'type': 'product_listing', 'category': 'filter'},
            'sort_selection': {'name': 'Sort Products', 'url': '/products/shoes?sort=price_low', 'type': 'product_listing', 'category': 'sort'},
            'search_query': {'name': 'Search', 'url': '/search?q=basketball', 'type': 'search', 'category': 'search'},
            'search_refinement': {'name': 'Search Refinement', 'url': '/search?q=basketball+shoes', 'type': 'search', 'category': 'search'},
            'brand_page_visit': {'name': 'Brand Page', 'url': '/brand/adidas', 'type': 'marketing', 'category': 'brand'},
            'sale_section_browse': {'name': 'Sale Section', 'url': '/sale', 'type': 'marketing', 'category': 'promotion'},
            'new_arrivals_view': {'name': 'New Arrivals', 'url': '/new-arrivals', 'type': 'marketing', 'category': 'browse'},
            'trending_products_view': {'name': 'Trending Products', 'url': '/trending', 'type': 'marketing', 'category': 'browse'},
            'size_guide_view': {'name': 'Size Guide', 'url': '/size-guide', 'type': 'support', 'category': 'guide'},
            
            # Product Interaction
            'product_detail_view': {'name': 'Product Details', 'url': '/product/nike-air-max-270', 'type': 'product_detail', 'category': 'product'},
            'product_image_zoom': {'name': 'Image Zoom', 'url': '/product/nike-air-max-270#gallery', 'type': 'product_detail', 'category': 'product'},
            'size_selection': {'name': 'Size Selection', 'url': '/product/nike-air-max-270#size', 'type': 'product_detail', 'category': 'product'},
            'color_selection': {'name': 'Color Selection', 'url': '/product/nike-air-max-270#color', 'type': 'product_detail', 'category': 'product'},
            'product_video_play': {'name': 'Product Video', 'url': '/product/nike-air-max-270#video', 'type': 'product_detail', 'category': 'media'},
            'review_section_view': {'name': 'Product Reviews', 'url': '/product/nike-air-max-270#reviews', 'type': 'product_detail', 'category': 'social_proof'},
            'qa_section_view': {'name': 'Q&A Section', 'url': '/product/nike-air-max-270#qa', 'type': 'product_detail', 'category': 'support'},
            'size_chart_view': {'name': 'Size Chart', 'url': '/product/nike-air-max-270#size-chart', 'type': 'product_detail', 'category': 'guide'},
            'product_comparison': {'name': 'Product Comparison', 'url': '/compare', 'type': 'tools', 'category': 'comparison'},
            'related_products_view': {'name': 'Related Products', 'url': '/product/nike-air-max-270#related', 'type': 'product_detail', 'category': 'cross_sell'},
            'recently_viewed_check': {'name': 'Recently Viewed', 'url': '/recently-viewed', 'type': 'account', 'category': 'personalization'},
            
            # Cart Interactions
            'add_to_cart': {'name': 'Add to Cart', 'url': '/cart/add', 'type': 'ecommerce', 'category': 'cart'},
            'cart_view': {'name': 'Shopping Cart', 'url': '/cart', 'type': 'ecommerce', 'category': 'cart'},
            'quantity_update': {'name': 'Update Quantity', 'url': '/cart/update', 'type': 'ecommerce', 'category': 'cart'},
            'remove_from_cart': {'name': 'Remove from Cart', 'url': '/cart/remove', 'type': 'ecommerce', 'category': 'cart'},
            'save_for_later': {'name': 'Save for Later', 'url': '/cart/save-later', 'type': 'ecommerce', 'category': 'cart'},
            'cart_abandonment_recovery': {'name': 'Cart Recovery', 'url': '/cart/recovery', 'type': 'ecommerce', 'category': 'recovery'},
            'promo_code_entry': {'name': 'Promo Code', 'url': '/cart/promo', 'type': 'ecommerce', 'category': 'promotion'},
            'shipping_calculator_use': {'name': 'Shipping Calculator', 'url': '/cart/shipping', 'type': 'tools', 'category': 'shipping'},
            'cart_share': {'name': 'Share Cart', 'url': '/cart/share', 'type': 'social', 'category': 'sharing'},
            
            # Checkout Process
            'checkout_initiation': {'name': 'Start Checkout', 'url': '/checkout', 'type': 'ecommerce', 'category': 'checkout'},
            'shipping_info_entry': {'name': 'Shipping Info', 'url': '/checkout/shipping', 'type': 'ecommerce', 'category': 'checkout'},
            'billing_info_entry': {'name': 'Billing Info', 'url': '/checkout/billing', 'type': 'ecommerce', 'category': 'checkout'},
            'payment_method_selection': {'name': 'Payment Method', 'url': '/checkout/payment', 'type': 'ecommerce', 'category': 'checkout'},
            'order_review': {'name': 'Order Review', 'url': '/checkout/review', 'type': 'ecommerce', 'category': 'checkout'},
            'purchase_completion': {'name': 'Purchase Complete', 'url': '/checkout/complete', 'type': 'ecommerce', 'category': 'purchase'},
            'order_confirmation_view': {'name': 'Order Confirmation', 'url': '/order/confirmation', 'type': 'ecommerce', 'category': 'confirmation'},
            'receipt_email_open': {'name': 'Receipt Email', 'url': '/email/receipt', 'type': 'email', 'category': 'confirmation'},
            
            # Account Management
            'account_dashboard_view': {'name': 'Account Dashboard', 'url': '/my-account', 'type': 'account', 'category': 'account'},
            'order_history_view': {'name': 'Order History', 'url': '/my-account/orders', 'type': 'account', 'category': 'account'},
            'profile_update': {'name': 'Update Profile', 'url': '/my-account/profile', 'type': 'account', 'category': 'account'},
            'address_book_management': {'name': 'Address Book', 'url': '/my-account/addresses', 'type': 'account', 'category': 'account'},
            'payment_methods_management': {'name': 'Payment Methods', 'url': '/my-account/payments', 'type': 'account', 'category': 'account'},
            'preferences_update': {'name': 'Preferences', 'url': '/my-account/preferences', 'type': 'account', 'category': 'account'},
            'subscription_management': {'name': 'Subscriptions', 'url': '/my-account/subscriptions', 'type': 'account', 'category': 'subscription'},
            'loyalty_points_check': {'name': 'Loyalty Points', 'url': '/my-account/loyalty', 'type': 'account', 'category': 'loyalty'},
            
            # Wishlist & Favorites
            'wishlist_view': {'name': 'Wishlist', 'url': '/wishlist', 'type': 'wishlist', 'category': 'wishlist'},
            'add_to_wishlist': {'name': 'Add to Wishlist', 'url': '/wishlist/add', 'type': 'wishlist', 'category': 'wishlist'},
            'remove_from_wishlist': {'name': 'Remove from Wishlist', 'url': '/wishlist/remove', 'type': 'wishlist', 'category': 'wishlist'},
            'wishlist_share': {'name': 'Share Wishlist', 'url': '/wishlist/share', 'type': 'social', 'category': 'sharing'},
            'move_to_cart_from_wishlist': {'name': 'Wishlist to Cart', 'url': '/wishlist/move-to-cart', 'type': 'wishlist', 'category': 'conversion'},
            'favorites_organization': {'name': 'Organize Favorites', 'url': '/wishlist/organize', 'type': 'wishlist', 'category': 'organization'},
            
            # Reviews & Social
            'review_submission': {'name': 'Submit Review', 'url': '/review/submit', 'type': 'social', 'category': 'review'},
            'review_reading': {'name': 'Read Reviews', 'url': '/reviews', 'type': 'social', 'category': 'social_proof'},
            'rating_submission': {'name': 'Submit Rating', 'url': '/rating/submit', 'type': 'social', 'category': 'rating'},
            'photo_review_upload': {'name': 'Photo Review', 'url': '/review/photo', 'type': 'social', 'category': 'ugc'},
            'social_share': {'name': 'Social Share', 'url': '/share', 'type': 'social', 'category': 'sharing'},
            'referral_program_use': {'name': 'Referral Program', 'url': '/referral', 'type': 'marketing', 'category': 'referral'},
            'user_generated_content_view': {'name': 'User Content', 'url': '/community', 'type': 'social', 'category': 'ugc'},
            
            # Support Touchpoints
            'help_center_visit': {'name': 'Help Center', 'url': '/help', 'type': 'support', 'category': 'support'},
            'faq_browse': {'name': 'FAQ', 'url': '/faq', 'type': 'support', 'category': 'support'},
            'live_chat_initiate': {'name': 'Live Chat', 'url': '/chat', 'type': 'support', 'category': 'support'},
            'contact_form_submission': {'name': 'Contact Form', 'url': '/contact', 'type': 'support', 'category': 'support'},
            'return_policy_view': {'name': 'Return Policy', 'url': '/returns', 'type': 'support', 'category': 'policy'},
            'shipping_info_view': {'name': 'Shipping Info', 'url': '/shipping', 'type': 'support', 'category': 'policy'},
            'size_exchange_request': {'name': 'Size Exchange', 'url': '/exchange', 'type': 'support', 'category': 'returns'},
            'order_tracking': {'name': 'Track Order', 'url': '/track', 'type': 'support', 'category': 'tracking'},
            'customer_service_call': {'name': 'Customer Service', 'url': '/support/call', 'type': 'support', 'category': 'phone'},
            'chatbot_interaction': {'name': 'Chatbot', 'url': '/chatbot', 'type': 'support', 'category': 'automation'},
            
            # Mobile Specific
            'app_notification_click': {'name': 'App Notification', 'url': '/app/notification', 'type': 'mobile', 'category': 'notification'},
            'barcode_scan': {'name': 'Barcode Scan', 'url': '/app/scan', 'type': 'mobile', 'category': 'scan'},
            'store_locator_use': {'name': 'Store Locator', 'url': '/stores', 'type': 'mobile', 'category': 'location'},
            'mobile_exclusive_offer_view': {'name': 'Mobile Offer', 'url': '/app/exclusive', 'type': 'mobile', 'category': 'promotion'},
            'push_notification_settings': {'name': 'Push Settings', 'url': '/app/settings/notifications', 'type': 'mobile', 'category': 'settings'},
            'mobile_payment_setup': {'name': 'Mobile Payment', 'url': '/app/payment', 'type': 'mobile', 'category': 'payment'},
            'offline_wishlist_sync': {'name': 'Offline Sync', 'url': '/app/sync', 'type': 'mobile', 'category': 'sync'},
            
            # Promotional
            'coupon_code_search': {'name': 'Coupon Search', 'url': '/coupons', 'type': 'promotion', 'category': 'promotion'},
            'sale_banner_click': {'name': 'Sale Banner', 'url': '/sale/flash', 'type': 'promotion', 'category': 'promotion'},
            'loyalty_program_join': {'name': 'Join Loyalty', 'url': '/loyalty/join', 'type': 'loyalty', 'category': 'loyalty'},
            'email_signup': {'name': 'Email Signup', 'url': '/newsletter', 'type': 'marketing', 'category': 'signup'},
            'sms_signup': {'name': 'SMS Signup', 'url': '/sms-alerts', 'type': 'marketing', 'category': 'signup'},
            'flash_sale_participation': {'name': 'Flash Sale', 'url': '/flash-sale', 'type': 'promotion', 'category': 'flash_sale'},
            'seasonal_promotion_view': {'name': 'Seasonal Sale', 'url': '/seasonal', 'type': 'promotion', 'category': 'seasonal'},
            'bundle_offer_view': {'name': 'Bundle Offer', 'url': '/bundles', 'type': 'promotion', 'category': 'bundle'},
            
            # Cross-selling
            'recommended_products_view': {'name': 'Recommended Products', 'url': '/recommendations', 'type': 'personalization', 'category': 'recommendation'},
            'frequently_bought_together': {'name': 'Frequently Bought Together', 'url': '/product/bundle', 'type': 'cross_sell', 'category': 'bundle'},
            'personalized_recommendations': {'name': 'Personalized Recs', 'url': '/for-you', 'type': 'personalization', 'category': 'personalization'},
            'category_upsell_view': {'name': 'Category Upsell', 'url': '/category/premium', 'type': 'upsell', 'category': 'upsell'},
            'accessory_suggestions': {'name': 'Accessory Suggestions', 'url': '/accessories', 'type': 'cross_sell', 'category': 'accessories'},
            'outfit_completion_suggestions': {'name': 'Complete the Look', 'url': '/outfit-builder', 'type': 'cross_sell', 'category': 'styling'},
            
            # Exits
            'logout': {'name': 'Logout', 'url': '/logout', 'type': 'authentication', 'category': 'exit'},
            'session_timeout': {'name': 'Session Timeout', 'url': '/timeout', 'type': 'system', 'category': 'exit'},
            'navigation_away': {'name': 'Navigate Away', 'url': '/external', 'type': 'system', 'category': 'exit'},
            'app_background': {'name': 'App Background', 'url': '/app/background', 'type': 'mobile', 'category': 'exit'},
            'browser_close': {'name': 'Browser Close', 'url': '/close', 'type': 'system', 'category': 'exit'},
            'checkout_abandonment': {'name': 'Checkout Abandonment', 'url': '/checkout/abandon', 'type': 'ecommerce', 'category': 'abandonment'}
        }
        
        # Sporting goods product data
        product_names = [
            'Nike Air Max 270', 'Adidas Ultraboost 22', 'Under Armour HOVR Phantom',
            'New Balance Fresh Foam X', 'ASICS Gel-Kayano 29', 'Brooks Ghost 15',
            'Nike Dri-FIT Training Shirt', 'Adidas Climalite Tank Top', 'Lululemon Swiftly Tech',
            'Under Armour HeatGear Leggings', 'Nike Pro Shorts', 'Adidas 3-Stripes Track Pants',
            'Patagonia Better Sweater', 'The North Face Venture Jacket', 'Columbia Flash Forward Windbreaker',
            'Wilson Tennis Racket Pro Staff', 'Spalding Basketball Official Size', 'Callaway Golf Driver',
            'Yeti Water Bottle Rambler', 'Hydro Flask Standard Mouth', 'Nike Training Gloves',
            'Fitbit Charge 5', 'Apple Watch Series 8', 'Garmin Forerunner 955'
        ]
        
        product_categories = [
            'running_shoes', 'training_shoes', 'basketball_shoes', 'tennis_shoes',
            'athletic_tops', 'athletic_bottoms', 'outerwear', 'swimwear',
            'team_sports', 'outdoor_recreation', 'fitness_accessories', 'technology'
        ]
        
        product_brands = [
            'Nike', 'Adidas', 'Under Armour', 'New Balance', 'ASICS', 'Brooks',
            'Lululemon', 'Patagonia', 'The North Face', 'Columbia', 'Reebok',
            'Puma', 'Wilson', 'Spalding', 'Callaway', 'Yeti', 'Hydro Flask'
        ]
        
        customer_segments = [
            'casual_fitness', 'serious_athlete', 'weekend_warrior', 'team_sports_player',
            'outdoor_enthusiast', 'fashion_focused', 'budget_conscious', 'premium_buyer'
        ]
        
        sport_preferences = [
            'running', 'basketball', 'tennis', 'soccer', 'golf', 'hiking',
            'yoga', 'crossfit', 'swimming', 'cycling', 'football', 'baseball'
        ]
        
        size_preferences = ['XS', 'S', 'M', 'L', 'XL', 'XXL', '6', '7', '8', '9', '10', '11', '12']
        
        payment_methods = ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay', 'klarna', 'afterpay']
        shipping_methods = ['standard', 'express', 'overnight', 'store_pickup', 'curbside_pickup']
        
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
            'youtube', 'tiktok', 'pinterest', 'influencer', 'affiliate'
        ]
        mediums = ['organic', 'cpc', 'email', 'social', 'referral', 'direct', 'display', 'video', 'influencer']
        
        # Generate consistent user profile for this journey
        visitor_id = str(uuid.uuid4())
        customer_id = str(uuid.uuid4())
        customer_segment = random.choice(customer_segments)
        sport_preference = random.choice(sport_preferences)
        size_preference = random.choice(size_preferences)
        
        clv_tiers = ['low', 'medium', 'high', 'premium']
        customer_lifetime_value_tier = random.choice(clv_tiers)
        loyalty_member = random.choice(['yes', 'no', 'pending'])
        
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
        
        # Add some randomization - 25% chance to add extra cross-category events
        if random.random() < 0.25:
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
        is_mobile_app = is_mobile and random.random() < 0.3
        if is_mobile_app:
            channel = 'mobile_app' if device == 'Mobile' else 'tablet_app'
        else:
            channel = f"web_{device.lower()}"
        
        # Campaign attribution (consistent for the journey)
        has_campaign = random.random() < 0.40
        campaign_id = str(uuid.uuid4()) if has_campaign else None
        traffic_source = random.choice(traffic_sources) if has_campaign else 'direct'
        medium = random.choice(mediums) if has_campaign else 'direct'
        referrer_domain = fake.domain_name() if traffic_source == 'referral' else None
        
        # Generate journey start time
        journey_start = datetime.now() - timedelta(
            days=random.randint(0, 90),
            hours=random.randint(6, 23),
            minutes=random.randint(0, 59)
        )
        
        # Site section mapping
        site_section_mapping = {
            'marketing': 'Marketing & Promotions',
            'authentication': 'Account & Login',
            'product_listing': 'Product Catalog',
            'product_detail': 'Product Details',
            'ecommerce': 'Shopping & Checkout',
            'account': 'My Account',
            'wishlist': 'Wishlist & Favorites',
            'social': 'Community & Reviews',
            'support': 'Customer Support',
            'mobile': 'Mobile Experience',
            'tools': 'Shopping Tools',
            'search': 'Search & Discovery'
        }
        
        # Generate events for the journey
        previous_url = None
        converted = False
        total_order_value = 0
        
        for i, event_type in enumerate(event_sequence):
            # Calculate event timestamp with realistic gaps
            if i == 0:
                event_timestamp = journey_start
            else:
                # Variable time gaps based on event category
                prev_category = event_details.get(event_sequence[i-1], {}).get('category', '')
                curr_category = event_details.get(event_type, {}).get('category', '')
                
                if prev_category == curr_category:
                    gap_seconds = random.randint(10, 90)  # 10 seconds to 1.5 minutes for related actions
                elif curr_category == 'checkout':
                    gap_seconds = random.randint(30, 180)  # Longer for checkout steps
                else:
                    gap_seconds = random.randint(30, 300)  # 30 seconds to 5 minutes for category changes
                
                event_timestamp = previous_timestamp + timedelta(seconds=gap_seconds)
            
            previous_timestamp = event_timestamp
            
            # Get event information
            event_info = event_details.get(event_type, {
                'name': event_type.replace('_', ' ').title(),
                'url': f'/{event_type.replace("_", "-")}',
                'type': 'general',
                'category': 'other'
            })
            
            # Product and pricing details
            product_name = random.choice(product_names) if event_info['category'] in ['product', 'cart', 'checkout', 'purchase'] else None
            product_category = random.choice(product_categories) if product_name else None
            product_brand = random.choice(product_brands) if product_name else None
            product_price = round(random.uniform(15, 300), 2) if product_name else None
            quantity = random.randint(1, 3) if product_name else None
            discount_amount = round(product_price * random.uniform(0, 0.3), 2) if product_price and random.random() < 0.3 else None
            
            payment_method = random.choice(payment_methods) if event_type == 'payment_method_selection' else None
            shipping_method = random.choice(shipping_methods) if event_type == 'shipping_info_entry' else None
            
            # Order value and revenue calculation
            order_value = None
            revenue_impact = None
            
            # Determine if this is a conversion event
            conversion_events = ['purchase_completion', 'order_confirmation_view']
            is_conversion_event = event_type in conversion_events
            
            if is_conversion_event and random.random() < journey_template['conversion_rate']:
                converted = True
                if journey_template['revenue_range'][1] > 0:
                    order_value = round(random.uniform(*journey_template['revenue_range']), 2)
                    revenue_impact = order_value
                    total_order_value = order_value
            
            # Custom events with explicit names (counts)
            product_views = 1 if event_info['category'] in ['product', 'browse'] else 0
            add_to_cart_events = 1 if event_type == 'add_to_cart' else 0
            purchase_events = 1 if event_type == 'purchase_completion' else 0
            search_events = 1 if event_info['category'] == 'search' else 0
            wishlist_additions = 1 if event_type == 'add_to_wishlist' else 0
            
            # Page interaction metrics
            time_on_page = random.randint(5, 600)  # 5 seconds to 10 minutes
            scroll_depth = random.randint(10, 100)  # Percentage
            clicks_on_page = random.randint(0, 25)
            page_load_time = random.randint(100, 4000)  # milliseconds
            
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
                
                # Ecommerce specific fields
                product_name,  # product_name
                product_category,  # product_category
                product_brand,  # product_brand
                product_price,  # product_price
                order_value,  # order_value
                quantity,  # quantity
                discount_amount,  # discount_amount
                payment_method,  # payment_method
                shipping_method,  # shipping_method
                
                # Campaign/Marketing (consistent for journey)
                campaign_id,  # campaign_id
                traffic_source,  # traffic_source
                medium,  # medium
                referrer_domain,  # referrer_domain
                
                # Custom dimensions with explicit names (consistent for journey)
                customer_segment,  # customer_segment
                customer_lifetime_value_tier,  # customer_lifetime_value_tier
                sport_preference,  # sport_preference
                size_preference,  # size_preference
                loyalty_member,  # loyalty_program_member
                
                # Custom events with explicit names
                product_views,  # product_views
                add_to_cart_events,  # add_to_cart_events
                purchase_events,  # purchase_events
                search_events,  # search_events
                wishlist_additions,  # wishlist_additions
                
                # Additional context
                is_mobile_app,  # is_mobile_app
                page_load_time,  # page_load_time_ms
                is_conversion_event and converted,  # conversion_flag
                revenue_impact  # revenue_impact
            )
            
            # Set previous URL for next iteration
            previous_url = event_info['url']
$$;

CREATE OR REPLACE TABLE retail_event_stream AS
SELECT e.*
FROM TABLE(GENERATOR(ROWCOUNT => 1000000)) g
CROSS JOIN TABLE(generate_ecommerce_journey()) e;

select * from retail_event_stream;