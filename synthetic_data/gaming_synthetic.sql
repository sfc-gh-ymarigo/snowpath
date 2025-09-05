CREATE OR REPLACE FUNCTION generate_gaming_journey()
RETURNS TABLE (
    -- Core identifiers
    event_id STRING,
    user_id STRING,
    player_id STRING,
    session_id STRING,
    
    -- Event details
    event_timestamp TIMESTAMP,
    event_type STRING,
    event_category STRING,
    event_action STRING,
    event_label STRING,
    
    -- Game/Platform information
    game_title STRING,
    game_mode STRING,
    platform STRING,
    game_version STRING,
    level_name STRING,
    
    -- Technical details
    device_type STRING,
    operating_system STRING,
    client_version STRING,
    connection_type STRING,
    fps_average INT,
    ping_ms INT,
    
    -- Geographic data
    country STRING,
    region STRING,
    timezone STRING,
    
    -- Gameplay metrics
    session_duration_minutes INT,
    actions_per_minute INT,
    score_achieved INT,
    
    -- Game-specific fields
    character_class STRING,
    character_level INT,
    current_xp INT,
    currency_earned INT,
    currency_spent INT,
    items_collected STRING,
    achievement_unlocked STRING,
    
    -- Store/Monetization fields
    item_purchased STRING,
    item_category STRING,
    item_rarity STRING,
    purchase_price NUMBER(10,2),
    currency_type STRING,
    total_spent NUMBER(12,2),
    
    -- Player behavior dimensions
    player_segment STRING,
    spending_tier STRING,
    skill_level STRING,
    playtime_category STRING,
    social_activity_level STRING,
    
    -- Custom gaming events
    level_completions INT,
    deaths_count INT,
    kills_count INT,
    items_used INT,
    social_interactions INT,
    
    -- Additional context
    is_premium_player BOOLEAN,
    is_first_session BOOLEAN,
    conversion_flag BOOLEAN,
    revenue_impact NUMBER(12,2)
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
        # Define shared event pools for gaming experiences
        shared_events = {
            'session_start': [
                'game_launch', 'login_success', 'main_menu_enter', 'tutorial_start',
                'continue_game', 'new_game_start', 'profile_load', 'settings_check'
            ],
            'gameplay_core': [
                'level_start', 'level_complete', 'level_failed', 'checkpoint_reached',
                'boss_encounter', 'boss_defeated', 'mission_start', 'mission_complete',
                'quest_accepted', 'quest_completed', 'objective_completed'
            ],
            'character_progression': [
                'level_up', 'xp_gained', 'skill_unlocked', 'ability_upgraded',
                'stat_increased', 'character_created', 'class_selected', 'talent_point_spent',
                'prestige_reached', 'achievement_earned'
            ],
            'combat_events': [
                'enemy_killed', 'player_death', 'damage_dealt', 'damage_taken',
                'critical_hit', 'combo_executed', 'spell_cast', 'item_used_combat',
                'weapon_equipped', 'armor_equipped'
            ],
            'item_collection': [
                'item_found', 'treasure_opened', 'loot_collected', 'rare_drop',
                'crafting_material_found', 'currency_found', 'item_crafted',
                'equipment_upgraded', 'inventory_full', 'item_sold'
            ],
            'social_multiplayer': [
                'friend_added', 'party_joined', 'guild_joined', 'chat_message_sent',
                'voice_chat_started', 'player_invited', 'match_found', 'team_formed',
                'leaderboard_viewed', 'tournament_entered'
            ],
            'store_browsing': [
                'store_opened', 'category_browsed', 'item_previewed', 'item_details_viewed',
                'price_checked', 'bundle_viewed', 'sale_items_browsed', 'wishlist_viewed',
                'recommendations_viewed', 'search_store'
            ],
            'monetization': [
                'item_purchased', 'bundle_purchased', 'currency_purchased', 'premium_upgrade',
                'battle_pass_purchased', 'dlc_purchased', 'cosmetic_purchased',
                'booster_purchased', 'subscription_activated', 'gift_purchased'
            ],
            'customization': [
                'character_customized', 'outfit_changed', 'weapon_skin_applied',
                'base_decorated', 'avatar_updated', 'title_changed', 'emblem_selected',
                'emote_equipped', 'victory_pose_set', 'loadout_saved'
            ],
            'meta_progression': [
                'daily_quest_completed', 'weekly_challenge_finished', 'event_participated',
                'seasonal_reward_claimed', 'battle_pass_tier_unlocked', 'login_bonus_claimed',
                'milestone_reached', 'collection_completed', 'mastery_achieved'
            ],
            'session_end': [
                'game_paused', 'settings_accessed', 'save_game', 'logout',
                'session_timeout', 'connection_lost', 'game_closed', 'platform_exit'
            ]
        }
        
        # Define journey templates for gaming experiences
        journey_templates = {
            'new_player_onboarding': {
                'primary_goal': 'tutorial_completion',
                'base_flow': [
                    ('session_start', 1),
                    ('gameplay_core', random.randint(3, 6)),
                    ('character_progression', random.randint(2, 4)),
                    ('combat_events', random.randint(2, 5)),
                    ('item_collection', random.randint(1, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.65,
                'revenue_range': (0, 10)
            },
            'casual_gaming_session': {
                'primary_goal': 'entertainment',
                'base_flow': [
                    ('session_start', 1),
                    ('gameplay_core', random.randint(2, 4)),
                    ('combat_events', random.randint(1, 3)),
                    ('item_collection', random.randint(1, 2)),
                    ('character_progression', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.85,
                'revenue_range': (0, 5)
            },
            'hardcore_gaming_session': {
                'primary_goal': 'progression',
                'base_flow': [
                    ('session_start', 1),
                    ('gameplay_core', random.randint(5, 10)),
                    ('combat_events', random.randint(4, 8)),
                    ('character_progression', random.randint(2, 5)),
                    ('item_collection', random.randint(2, 4)),
                    ('meta_progression', random.randint(1, 3)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.90,
                'revenue_range': (0, 50)
            },
            'competitive_multiplayer': {
                'primary_goal': 'ranking_improvement',
                'base_flow': [
                    ('session_start', 1),
                    ('social_multiplayer', random.randint(2, 4)),
                    ('gameplay_core', random.randint(3, 6)),
                    ('combat_events', random.randint(5, 10)),
                    ('character_progression', random.randint(1, 3)),
                    ('meta_progression', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.75,
                'revenue_range': (0, 25)
            },
            'shopping_spree': {
                'primary_goal': 'store_purchase',
                'base_flow': [
                    ('session_start', 1),
                    ('store_browsing', random.randint(3, 6)),
                    ('monetization', random.randint(1, 4)),
                    ('customization', random.randint(1, 3)),
                    ('gameplay_core', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.45,
                'revenue_range': (5, 100)
            },
            'social_gaming_session': {
                'primary_goal': 'social_interaction',
                'base_flow': [
                    ('session_start', 1),
                    ('social_multiplayer', random.randint(3, 6)),
                    ('gameplay_core', random.randint(2, 4)),
                    ('combat_events', random.randint(2, 5)),
                    ('customization', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.70,
                'revenue_range': (0, 20)
            },
            'event_participation': {
                'primary_goal': 'event_completion',
                'base_flow': [
                    ('session_start', 1),
                    ('meta_progression', random.randint(2, 4)),
                    ('gameplay_core', random.randint(3, 6)),
                    ('combat_events', random.randint(2, 5)),
                    ('item_collection', random.randint(1, 3)),
                    ('monetization', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.55,
                'revenue_range': (0, 30)
            },
            'whale_spending_session': {
                'primary_goal': 'high_value_purchase',
                'base_flow': [
                    ('session_start', 1),
                    ('store_browsing', random.randint(2, 4)),
                    ('monetization', random.randint(3, 7)),
                    ('customization', random.randint(2, 4)),
                    ('gameplay_core', random.randint(1, 3)),
                    ('character_progression', random.randint(1, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.80,
                'revenue_range': (50, 500)
            },
            'tutorial_dropout': {
                'primary_goal': 'early_exit',
                'base_flow': [
                    ('session_start', 1),
                    ('gameplay_core', random.randint(1, 3)),
                    ('combat_events', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.05,
                'revenue_range': (0, 0)
            },
            'return_player_session': {
                'primary_goal': 're_engagement',
                'base_flow': [
                    ('session_start', 1),
                    ('meta_progression', random.randint(1, 2)),
                    ('store_browsing', random.randint(0, 2)),
                    ('gameplay_core', random.randint(2, 5)),
                    ('character_progression', random.randint(1, 3)),
                    ('monetization', random.randint(0, 2)),
                    ('session_end', 1)
                ],
                'conversion_rate': 0.40,
                'revenue_range': (0, 40)
            }
        }
        
        # Detailed event mappings for gaming
        event_details = {
            # Session Start
            'game_launch': {'name': 'Game Launch', 'url': '/game/launch', 'type': 'system', 'category': 'session'},
            'login_success': {'name': 'Login Success', 'url': '/login/success', 'type': 'authentication', 'category': 'session'},
            'main_menu_enter': {'name': 'Main Menu', 'url': '/menu/main', 'type': 'navigation', 'category': 'session'},
            'tutorial_start': {'name': 'Tutorial Start', 'url': '/tutorial/start', 'type': 'onboarding', 'category': 'session'},
            'continue_game': {'name': 'Continue Game', 'url': '/game/continue', 'type': 'navigation', 'category': 'session'},
            'new_game_start': {'name': 'New Game', 'url': '/game/new', 'type': 'navigation', 'category': 'session'},
            'profile_load': {'name': 'Profile Load', 'url': '/profile/load', 'type': 'system', 'category': 'session'},
            'settings_check': {'name': 'Settings Check', 'url': '/settings', 'type': 'navigation', 'category': 'session'},
            
            # Gameplay Core
            'level_start': {'name': 'Level Start', 'url': '/level/start', 'type': 'gameplay', 'category': 'core'},
            'level_complete': {'name': 'Level Complete', 'url': '/level/complete', 'type': 'gameplay', 'category': 'core'},
            'level_failed': {'name': 'Level Failed', 'url': '/level/failed', 'type': 'gameplay', 'category': 'core'},
            'checkpoint_reached': {'name': 'Checkpoint Reached', 'url': '/checkpoint', 'type': 'gameplay', 'category': 'core'},
            'boss_encounter': {'name': 'Boss Encounter', 'url': '/boss/encounter', 'type': 'gameplay', 'category': 'core'},
            'boss_defeated': {'name': 'Boss Defeated', 'url': '/boss/defeated', 'type': 'gameplay', 'category': 'core'},
            'mission_start': {'name': 'Mission Start', 'url': '/mission/start', 'type': 'gameplay', 'category': 'core'},
            'mission_complete': {'name': 'Mission Complete', 'url': '/mission/complete', 'type': 'gameplay', 'category': 'core'},
            'quest_accepted': {'name': 'Quest Accepted', 'url': '/quest/accept', 'type': 'gameplay', 'category': 'core'},
            'quest_completed': {'name': 'Quest Completed', 'url': '/quest/complete', 'type': 'gameplay', 'category': 'core'},
            'objective_completed': {'name': 'Objective Complete', 'url': '/objective/complete', 'type': 'gameplay', 'category': 'core'},
            
            # Character Progression
            'level_up': {'name': 'Level Up', 'url': '/character/levelup', 'type': 'progression', 'category': 'character'},
            'xp_gained': {'name': 'XP Gained', 'url': '/character/xp', 'type': 'progression', 'category': 'character'},
            'skill_unlocked': {'name': 'Skill Unlocked', 'url': '/character/skill', 'type': 'progression', 'category': 'character'},
            'ability_upgraded': {'name': 'Ability Upgraded', 'url': '/character/ability', 'type': 'progression', 'category': 'character'},
            'stat_increased': {'name': 'Stat Increased', 'url': '/character/stats', 'type': 'progression', 'category': 'character'},
            'character_created': {'name': 'Character Created', 'url': '/character/create', 'type': 'progression', 'category': 'character'},
            'class_selected': {'name': 'Class Selected', 'url': '/character/class', 'type': 'progression', 'category': 'character'},
            'talent_point_spent': {'name': 'Talent Point Spent', 'url': '/character/talent', 'type': 'progression', 'category': 'character'},
            'prestige_reached': {'name': 'Prestige Reached', 'url': '/character/prestige', 'type': 'progression', 'category': 'character'},
            'achievement_earned': {'name': 'Achievement Earned', 'url': '/achievement', 'type': 'progression', 'category': 'character'},
            
            # Combat Events
            'enemy_killed': {'name': 'Enemy Killed', 'url': '/combat/kill', 'type': 'combat', 'category': 'combat'},
            'player_death': {'name': 'Player Death', 'url': '/combat/death', 'type': 'combat', 'category': 'combat'},
            'damage_dealt': {'name': 'Damage Dealt', 'url': '/combat/damage_out', 'type': 'combat', 'category': 'combat'},
            'damage_taken': {'name': 'Damage Taken', 'url': '/combat/damage_in', 'type': 'combat', 'category': 'combat'},
            'critical_hit': {'name': 'Critical Hit', 'url': '/combat/critical', 'type': 'combat', 'category': 'combat'},
            'combo_executed': {'name': 'Combo Executed', 'url': '/combat/combo', 'type': 'combat', 'category': 'combat'},
            'spell_cast': {'name': 'Spell Cast', 'url': '/combat/spell', 'type': 'combat', 'category': 'combat'},
            'item_used_combat': {'name': 'Item Used in Combat', 'url': '/combat/item', 'type': 'combat', 'category': 'combat'},
            'weapon_equipped': {'name': 'Weapon Equipped', 'url': '/equipment/weapon', 'type': 'equipment', 'category': 'combat'},
            'armor_equipped': {'name': 'Armor Equipped', 'url': '/equipment/armor', 'type': 'equipment', 'category': 'combat'},
            
            # Item Collection
            'item_found': {'name': 'Item Found', 'url': '/loot/item', 'type': 'collection', 'category': 'items'},
            'treasure_opened': {'name': 'Treasure Opened', 'url': '/loot/treasure', 'type': 'collection', 'category': 'items'},
            'loot_collected': {'name': 'Loot Collected', 'url': '/loot/collect', 'type': 'collection', 'category': 'items'},
            'rare_drop': {'name': 'Rare Drop', 'url': '/loot/rare', 'type': 'collection', 'category': 'items'},
            'crafting_material_found': {'name': 'Crafting Material', 'url': '/loot/material', 'type': 'collection', 'category': 'items'},
            'currency_found': {'name': 'Currency Found', 'url': '/loot/currency', 'type': 'collection', 'category': 'items'},
            'item_crafted': {'name': 'Item Crafted', 'url': '/crafting/craft', 'type': 'crafting', 'category': 'items'},
            'equipment_upgraded': {'name': 'Equipment Upgraded', 'url': '/crafting/upgrade', 'type': 'crafting', 'category': 'items'},
            'inventory_full': {'name': 'Inventory Full', 'url': '/inventory/full', 'type': 'inventory', 'category': 'items'},
            'item_sold': {'name': 'Item Sold', 'url': '/inventory/sell', 'type': 'economy', 'category': 'items'},
            
            # Social & Multiplayer
            'friend_added': {'name': 'Friend Added', 'url': '/social/friend', 'type': 'social', 'category': 'social'},
            'party_joined': {'name': 'Party Joined', 'url': '/social/party', 'type': 'social', 'category': 'social'},
            'guild_joined': {'name': 'Guild Joined', 'url': '/social/guild', 'type': 'social', 'category': 'social'},
            'chat_message_sent': {'name': 'Chat Message', 'url': '/social/chat', 'type': 'social', 'category': 'social'},
            'voice_chat_started': {'name': 'Voice Chat', 'url': '/social/voice', 'type': 'social', 'category': 'social'},
            'player_invited': {'name': 'Player Invited', 'url': '/social/invite', 'type': 'social', 'category': 'social'},
            'match_found': {'name': 'Match Found', 'url': '/matchmaking/found', 'type': 'matchmaking', 'category': 'social'},
            'team_formed': {'name': 'Team Formed', 'url': '/matchmaking/team', 'type': 'matchmaking', 'category': 'social'},
            'leaderboard_viewed': {'name': 'Leaderboard Viewed', 'url': '/leaderboard', 'type': 'competitive', 'category': 'social'},
            'tournament_entered': {'name': 'Tournament Entered', 'url': '/tournament/enter', 'type': 'competitive', 'category': 'social'},
            
            # Store Browsing
            'store_opened': {'name': 'Store Opened', 'url': '/store', 'type': 'store', 'category': 'monetization'},
            'category_browsed': {'name': 'Category Browsed', 'url': '/store/category', 'type': 'store', 'category': 'monetization'},
            'item_previewed': {'name': 'Item Previewed', 'url': '/store/preview', 'type': 'store', 'category': 'monetization'},
            'item_details_viewed': {'name': 'Item Details', 'url': '/store/details', 'type': 'store', 'category': 'monetization'},
            'price_checked': {'name': 'Price Checked', 'url': '/store/price', 'type': 'store', 'category': 'monetization'},
            'bundle_viewed': {'name': 'Bundle Viewed', 'url': '/store/bundle', 'type': 'store', 'category': 'monetization'},
            'sale_items_browsed': {'name': 'Sale Items', 'url': '/store/sale', 'type': 'store', 'category': 'monetization'},
            'wishlist_viewed': {'name': 'Wishlist Viewed', 'url': '/store/wishlist', 'type': 'store', 'category': 'monetization'},
            'recommendations_viewed': {'name': 'Recommendations', 'url': '/store/recommended', 'type': 'store', 'category': 'monetization'},
            'search_store': {'name': 'Store Search', 'url': '/store/search', 'type': 'store', 'category': 'monetization'},
            
            # Monetization
            'item_purchased': {'name': 'Item Purchased', 'url': '/store/purchase', 'type': 'purchase', 'category': 'monetization'},
            'bundle_purchased': {'name': 'Bundle Purchased', 'url': '/store/bundle_buy', 'type': 'purchase', 'category': 'monetization'},
            'currency_purchased': {'name': 'Currency Purchased', 'url': '/store/currency', 'type': 'purchase', 'category': 'monetization'},
            'premium_upgrade': {'name': 'Premium Upgrade', 'url': '/store/premium', 'type': 'purchase', 'category': 'monetization'},
            'battle_pass_purchased': {'name': 'Battle Pass', 'url': '/store/battlepass', 'type': 'purchase', 'category': 'monetization'},
            'dlc_purchased': {'name': 'DLC Purchased', 'url': '/store/dlc', 'type': 'purchase', 'category': 'monetization'},
            'cosmetic_purchased': {'name': 'Cosmetic Purchased', 'url': '/store/cosmetic', 'type': 'purchase', 'category': 'monetization'},
            'booster_purchased': {'name': 'Booster Purchased', 'url': '/store/booster', 'type': 'purchase', 'category': 'monetization'},
            'subscription_activated': {'name': 'Subscription Active', 'url': '/store/subscription', 'type': 'purchase', 'category': 'monetization'},
            'gift_purchased': {'name': 'Gift Purchased', 'url': '/store/gift', 'type': 'purchase', 'category': 'monetization'},
            
            # Customization
            'character_customized': {'name': 'Character Customized', 'url': '/customize/character', 'type': 'customization', 'category': 'customization'},
            'outfit_changed': {'name': 'Outfit Changed', 'url': '/customize/outfit', 'type': 'customization', 'category': 'customization'},
            'weapon_skin_applied': {'name': 'Weapon Skin Applied', 'url': '/customize/weapon', 'type': 'customization', 'category': 'customization'},
            'base_decorated': {'name': 'Base Decorated', 'url': '/customize/base', 'type': 'customization', 'category': 'customization'},
            'avatar_updated': {'name': 'Avatar Updated', 'url': '/customize/avatar', 'type': 'customization', 'category': 'customization'},
            'title_changed': {'name': 'Title Changed', 'url': '/customize/title', 'type': 'customization', 'category': 'customization'},
            'emblem_selected': {'name': 'Emblem Selected', 'url': '/customize/emblem', 'type': 'customization', 'category': 'customization'},
            'emote_equipped': {'name': 'Emote Equipped', 'url': '/customize/emote', 'type': 'customization', 'category': 'customization'},
            'victory_pose_set': {'name': 'Victory Pose Set', 'url': '/customize/victory', 'type': 'customization', 'category': 'customization'},
            'loadout_saved': {'name': 'Loadout Saved', 'url': '/customize/loadout', 'type': 'customization', 'category': 'customization'},
            
            # Meta Progression
            'daily_quest_completed': {'name': 'Daily Quest Complete', 'url': '/meta/daily', 'type': 'meta_progression', 'category': 'meta'},
            'weekly_challenge_finished': {'name': 'Weekly Challenge', 'url': '/meta/weekly', 'type': 'meta_progression', 'category': 'meta'},
            'event_participated': {'name': 'Event Participated', 'url': '/meta/event', 'type': 'meta_progression', 'category': 'meta'},
            'seasonal_reward_claimed': {'name': 'Seasonal Reward', 'url': '/meta/seasonal', 'type': 'meta_progression', 'category': 'meta'},
            'battle_pass_tier_unlocked': {'name': 'Battle Pass Tier', 'url': '/meta/battlepass_tier', 'type': 'meta_progression', 'category': 'meta'},
            'login_bonus_claimed': {'name': 'Login Bonus', 'url': '/meta/login_bonus', 'type': 'meta_progression', 'category': 'meta'},
            'milestone_reached': {'name': 'Milestone Reached', 'url': '/meta/milestone', 'type': 'meta_progression', 'category': 'meta'},
            'collection_completed': {'name': 'Collection Complete', 'url': '/meta/collection', 'type': 'meta_progression', 'category': 'meta'},
            'mastery_achieved': {'name': 'Mastery Achieved', 'url': '/meta/mastery', 'type': 'meta_progression', 'category': 'meta'},
            
            # Session End
            'game_paused': {'name': 'Game Paused', 'url': '/game/pause', 'type': 'system', 'category': 'session_end'},
            'settings_accessed': {'name': 'Settings Accessed', 'url': '/settings', 'type': 'navigation', 'category': 'session_end'},
            'save_game': {'name': 'Game Saved', 'url': '/game/save', 'type': 'system', 'category': 'session_end'},
            'logout': {'name': 'Logout', 'url': '/logout', 'type': 'authentication', 'category': 'session_end'},
            'session_timeout': {'name': 'Session Timeout', 'url': '/timeout', 'type': 'system', 'category': 'session_end'},
            'connection_lost': {'name': 'Connection Lost', 'url': '/disconnect', 'type': 'system', 'category': 'session_end'},
            'game_closed': {'name': 'Game Closed', 'url': '/game/close', 'type': 'system', 'category': 'session_end'},
            'platform_exit': {'name': 'Platform Exit', 'url': '/platform/exit', 'type': 'system', 'category': 'session_end'}
        }
        
        # Gaming data
        game_titles = [
            'Epic Quest Chronicles', 'Battle Royale Arena', 'Space Marine Command',
            'Fantasy Realm Adventures', 'City Builder Tycoon', 'Racing Champions',
            'Puzzle Master Pro', 'Card Legends', 'Tower Defense Elite',
            'MMORPG Worlds', 'First Person Shooter', 'Strategy Empire',
            'Platform Hero', 'Fighting Tournament', 'Survival Island'
        ]
        
        game_modes = [
            'single_player', 'multiplayer', 'co_op', 'competitive', 'ranked',
            'casual', 'tutorial', 'practice', 'campaign', 'survival',
            'battle_royale', 'team_deathmatch', 'capture_flag', 'domination'
        ]
        
        platforms = [
            'PC_Steam', 'PC_Epic', 'PlayStation_5', 'PlayStation_4', 'Xbox_Series_X',
            'Xbox_One', 'Nintendo_Switch', 'iOS', 'Android', 'Web_Browser'
        ]
        
        character_classes = [
            'Warrior', 'Mage', 'Archer', 'Rogue', 'Paladin', 'Necromancer',
            'Healer', 'Tank', 'Support', 'Assassin', 'Berserker', 'Shaman'
        ]
        
        item_categories = [
            'weapons', 'armor', 'cosmetics', 'consumables', 'currency',
            'boosters', 'battle_pass', 'dlc', 'character_packs', 'emotes'
        ]
        
        item_rarities = ['common', 'uncommon', 'rare', 'epic', 'legendary', 'mythic']
        
        currency_types = ['gold', 'gems', 'coins', 'crystals', 'premium_currency', 'real_money']
        
        player_segments = [
            'new_player', 'casual_gamer', 'core_gamer', 'hardcore_gamer',
            'competitive_player', 'social_player', 'whale_spender', 'content_creator'
        ]
        
        spending_tiers = ['free_to_play', 'low_spender', 'moderate_spender', 'high_spender', 'whale']
        skill_levels = ['beginner', 'novice', 'intermediate', 'advanced', 'expert', 'pro']
        playtime_categories = ['casual', 'regular', 'frequent', 'heavy', 'addicted']
        social_activity_levels = ['solo', 'occasional', 'social', 'very_social', 'community_leader']
        
        device_types = ['Desktop', 'Mobile', 'Console', 'Tablet']
        operating_systems = [
            'Windows 11', 'Windows 10', 'macOS 14', 'macOS 13',
            'iOS 17', 'iOS 16', 'Android 14', 'Android 13',
            'PlayStation OS', 'Xbox OS', 'Nintendo OS'
        ]
        connection_types = ['wifi', 'ethernet', 'cellular_5g', 'cellular_4g', 'cellular_3g']
        
        # Generate consistent user profile for this journey
        user_id = str(uuid.uuid4())
        player_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        player_segment = random.choice(player_segments)
        spending_tier = random.choice(spending_tiers)
        skill_level = random.choice(skill_levels)
        playtime_category = random.choice(playtime_categories)
        social_activity_level = random.choice(social_activity_levels)
        
        # Consistent game and character data
        game_title = random.choice(game_titles)
        game_mode = random.choice(game_modes)
        platform = random.choice(platforms)
        game_version = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        character_class = random.choice(character_classes)
        character_level = random.randint(1, 100)
        current_xp = random.randint(0, 10000)
        
        # Technical details
        device_type = random.choice(device_types)
        os = random.choice(operating_systems)
        client_version = f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        connection_type = random.choice(connection_types)
        
        # Performance metrics
        fps_average = random.randint(30, 120)
        ping_ms = random.randint(10, 200)
        
        # Geographic data
        country = 'United States'
        region = fake.state()
        timezone = random.choice(['PST', 'MST', 'CST', 'EST'])
        
        # Premium status
        is_premium_player = spending_tier in ['high_spender', 'whale']
        is_first_session = random.random() < 0.1  # 10% are first sessions
        
        # Choose a journey template
        journey_name = random.choice(list(journey_templates.keys()))
        journey_template = journey_templates[journey_name]
        
        # Build the actual event sequence from the template
        event_sequence = []
        for event_category, count in journey_template['base_flow']:
            selected_events = random.sample(shared_events[event_category], min(count, len(shared_events[event_category])))
            event_sequence.extend(selected_events)
        
        # Add some randomization - 15% chance to add extra events
        if random.random() < 0.15:
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
        total_spent = 0
        session_duration = random.randint(5, 180)  # 5 to 180 minutes
        
        for i, event_type in enumerate(event_sequence):
            # Calculate event timestamp
            if i == 0:
                event_timestamp = journey_start
            else:
                # Gaming events happen more frequently
                gap_seconds = random.randint(10, 300)  # 10 seconds to 5 minutes
                event_timestamp = previous_timestamp + timedelta(seconds=gap_seconds)
            
            previous_timestamp = event_timestamp
            
            # Get event information
            event_info = event_details.get(event_type, {
                'name': event_type.replace('_', ' ').title(),
                'url': f'/{event_type.replace("_", "/")}',
                'type': 'general',
                'category': 'other'
            })
            
            # Level information
            level_names = [
                'Tutorial Zone', 'Forest of Beginnings', 'Dark Cave', 'Mountain Peak',
                'Fire Temple', 'Ice Cavern', 'Sky Castle', 'Final Boss Arena',
                'PvP Arena', 'Raid Dungeon', 'Training Ground', 'Hub World'
            ]
            level_name = random.choice(level_names) if event_info['category'] in ['core', 'combat'] else None
            
            # Gameplay metrics
            actions_per_minute = random.randint(5, 50)
            score_achieved = random.randint(0, 10000) if event_info['category'] in ['core', 'combat'] else None
            
            # Currency and items
            currency_earned = random.randint(0, 500) if event_info['category'] in ['core', 'items'] else None
            currency_spent = 0
            items_collected = None
            achievement_unlocked = None
            
            if event_type in ['item_found', 'loot_collected', 'rare_drop']:
                items_collected = f"{random.choice(['Sword', 'Shield', 'Potion', 'Gem', 'Scroll'])} of {random.choice(['Power', 'Speed', 'Strength', 'Magic', 'Wisdom'])}"
            
            if event_type == 'achievement_earned':
                achievement_unlocked = f"{random.choice(['First', 'Master', 'Elite', 'Legendary'])} {random.choice(['Fighter', 'Explorer', 'Collector', 'Survivor'])}"
            
            # Store purchases
            item_purchased = None
            item_category = None
            item_rarity = None
            purchase_price = None
            purchase_currency_type = None
            revenue_impact = None
            
            if event_info['category'] == 'monetization' and 'purchased' in event_type:
                item_category = random.choice(item_categories)
                item_rarity = random.choice(item_rarities)
                purchase_currency_type = random.choice(currency_types)
                
                # Price based on rarity and category
                base_prices = {
                    'common': 1, 'uncommon': 5, 'rare': 15, 'epic': 25, 'legendary': 50, 'mythic': 100
                }
                category_multipliers = {
                    'weapons': 2.00, 'armor': 1.50, 'cosmetics': 1.00, 'consumables': 0.50,
                    'currency': 1.00, 'boosters': 0.80, 'battle_pass': 3.00, 'dlc': 5.00
                }
                
                purchase_price = base_prices[item_rarity] * category_multipliers.get(item_category, 1.0)
                
                if purchase_currency_type == 'real_money':
                    purchase_price = round(purchase_price, 2)
                    revenue_impact = purchase_price
                    total_spent += purchase_price
                else:
                    purchase_price = int(purchase_price * 100)  # Convert to in-game currency
                    currency_spent = purchase_price
                
                item_purchased = f"{item_rarity.title()} {item_category.replace('_', ' ').title()}"
            
            # Conversion determination
            conversion_events = [
                'item_purchased', 'bundle_purchased', 'premium_upgrade', 'battle_pass_purchased'
            ]
            is_conversion_event = event_type in conversion_events
            
            if is_conversion_event and random.random() < journey_template['conversion_rate']:
                converted = True
            
            # Custom gaming events (counts)
            level_completions = 1 if event_type in ['level_complete', 'mission_complete', 'quest_completed'] else 0
            deaths_count = 1 if event_type == 'player_death' else 0
            kills_count = random.randint(0, 5) if event_type == 'enemy_killed' else 0
            items_used = 1 if 'item_used' in event_type or 'equipped' in event_type else 0
            social_interactions = 1 if event_info['category'] == 'social' else 0
            
            yield (
                # Core identifiers
                str(uuid.uuid4()),  # event_id
                user_id,  # user_id (consistent)
                player_id,  # player_id (consistent)
                session_id,  # session_id (consistent)
                
                # Event details
                event_timestamp,  # event_timestamp
                event_type,  # event_type
                event_info['category'],  # event_category
                event_type.replace('_', ' ').title(),  # event_action
                f"{journey_template['primary_goal']}_{event_type}",  # event_label
                
                # Game/Platform information
                game_title,  # game_title
                game_mode,  # game_mode
                platform,  # platform
                game_version,  # game_version
                level_name,  # level_name
                
                # Technical details
                device_type,  # device_type
                os,  # operating_system
                client_version,  # client_version
                connection_type,  # connection_type
                fps_average,  # fps_average
                ping_ms,  # ping_ms
                
                # Geographic data
                country,  # country
                region,  # region
                timezone,  # timezone
                
                # Gameplay metrics
                session_duration,  # session_duration_minutes
                actions_per_minute,  # actions_per_minute
                score_achieved,  # score_achieved
                
                # Game-specific fields
                character_class,  # character_class
                character_level,  # character_level
                current_xp,  # current_xp
                currency_earned,  # currency_earned
                currency_spent,  # currency_spent
                items_collected,  # items_collected
                achievement_unlocked,  # achievement_unlocked
                
                # Store/Monetization fields
                item_purchased,  # item_purchased
                item_category,  # item_category
                item_rarity,  # item_rarity
                purchase_price,  # purchase_price
                purchase_currency_type,  # currency_type
                total_spent,  # total_spent
                
                # Player behavior dimensions
                player_segment,  # player_segment
                spending_tier,  # spending_tier
                skill_level,  # skill_level
                playtime_category,  # playtime_category
                social_activity_level,  # social_activity_level
                
                # Custom gaming events
                level_completions,  # level_completions
                deaths_count,  # deaths_count
                kills_count,  # kills_count
                items_used,  # items_used
                social_interactions,  # social_interactions
                
                # Additional context
                is_premium_player,  # is_premium_player
                is_first_session,  # is_first_session
                is_conversion_event and converted,  # conversion_flag
                revenue_impact  # revenue_impact
            )
$$;

CREATE OR REPLACE TABLE gaming_event_stream AS
SELECT e.*
FROM TABLE(GENERATOR(ROWCOUNT => 1000000)) g
CROSS JOIN TABLE(generate_gaming_journey()) e;

select * from gaming_event_stream;
