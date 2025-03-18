import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from enum import Enum
import asyncio
import aiohttp
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from functools import lru_cache
import aiofiles
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyCategory(Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    LAND = "land"

@dataclass
class BaseProperty:
    property_category: PropertyCategory
    location: str = ""
    price: Dict[str, str] = None  # {rent: "", deposit: ""}
    area: Dict[str, str] = None   # {value: "", unit: ""}
    additional_details: List[str] = None
    contact: List[str] = None
    raw_text: str = ""

    def __post_init__(self):
        if self.price is None:
            self.price = {"rent": "", "deposit": ""}
        if self.area is None:
            self.area = {"value": "", "unit": ""}
        if self.additional_details is None:
            self.additional_details = []
        if self.contact is None:
            self.contact = []

    def to_dict(self) -> Dict:
        """Convert property to dictionary"""
        return {
            "property_category": self.property_category.value,
            "location": self.location,
            "price": self.price,
            "area": self.area,
            "additional_details": self.additional_details,
            "contact": self.contact,
            "raw_text": self.raw_text
        }

@dataclass
class ResidentialProperty(BaseProperty):
    configuration: str = ""  # 1BHK, 2BHK, etc.
    furnishing_status: str = ""  # Fully Furnished, Semi Furnished, Unfurnished
    floor: str = ""
    total_floors: str = ""
    amenities: List[str] = None
    preferred_tenants: List[str] = None  # Family, Bachelor, etc.
    parking: Dict[str, int] = None  # {two_wheeler: 0, four_wheeler: 0}

    def __post_init__(self):
        super().__post_init__()
        if self.amenities is None:
            self.amenities = []
        if self.preferred_tenants is None:
            self.preferred_tenants = []
        if self.parking is None:
            self.parking = {"two_wheeler": 0, "four_wheeler": 0}

    def to_dict(self) -> Dict:
        """Convert residential property to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "configuration": self.configuration,
            "furnishing_status": self.furnishing_status,
            "floor": self.floor,
            "total_floors": self.total_floors,
            "amenities": self.amenities,
            "preferred_tenants": self.preferred_tenants,
            "parking": self.parking
        })
        return base_dict

@dataclass
class CommercialProperty(BaseProperty):
    property_type: str = ""  # Shop, Office, etc.
    floor: str = ""
    furnishing_status: str = ""
    washroom: bool = False
    parking_available: bool = False
    road_facing: bool = False
    water_connection: bool = False
    suitable_for: List[str] = None  # Restaurant, Clinic, etc.

    def __post_init__(self):
        super().__post_init__()
        if self.suitable_for is None:
            self.suitable_for = []

    def to_dict(self) -> Dict:
        """Convert commercial property to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "property_type": self.property_type,
            "floor": self.floor,
            "furnishing_status": self.furnishing_status,
            "washroom": self.washroom,
            "parking_available": self.parking_available,
            "road_facing": self.road_facing,
            "water_connection": self.water_connection,
            "suitable_for": self.suitable_for
        })
        return base_dict

@dataclass
class LandProperty(BaseProperty):
    plot_type: str = ""  # Industrial, Residential, Agricultural
    road_width: str = ""
    boundaries: bool = False
    corner_plot: bool = False
    legal_restrictions: List[str] = None
    development_potential: List[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.legal_restrictions is None:
            self.legal_restrictions = []
        if self.development_potential is None:
            self.development_potential = []

    def to_dict(self) -> Dict:
        """Convert land property to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "plot_type": self.plot_type,
            "road_width": self.road_width,
            "boundaries": self.boundaries,
            "corner_plot": self.corner_plot,
            "legal_restrictions": self.legal_restrictions,
            "development_potential": self.development_potential
        })
        return base_dict

@dataclass
class RealEstateInquiry:
    requirements: str = ""
    preferred_locations: List[str] = None
    budget: Dict[str, str] = None  # {min: "", max: ""}
    property_type: str = ""  # residential, commercial, land
    contact_info: List[str] = None
    additional_requirements: List[str] = None
    raw_text: str = ""

    def __post_init__(self):
        if self.preferred_locations is None:
            self.preferred_locations = []
        if self.budget is None:
            self.budget = {"min": "", "max": ""}
        if self.contact_info is None:
            self.contact_info = []
        if self.additional_requirements is None:
            self.additional_requirements = []

    def to_dict(self) -> Dict:
        return {
            "requirements": self.requirements,
            "preferred_locations": self.preferred_locations,
            "budget": self.budget,
            "property_type": self.property_type,
            "contact_info": self.contact_info,
            "additional_requirements": self.additional_requirements,
            "raw_text": self.raw_text
        }

# First, let's create a Message container class
@dataclass
class WhatsAppMessage:
    timestamp: str
    sender: str
    raw_text: str
    properties: List[Union[ResidentialProperty, CommercialProperty, LandProperty]] = None
    inquiries: List[RealEstateInquiry] = None
    message_index: int = 0

    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.inquiries is None:
            self.inquiries = []

    def to_dict(self) -> Dict:
        # Basic message info
        basic_info = {
            "timestamp": self.timestamp,
            "sender": self.sender,
            "message_type": self._determine_message_type()
        }

        # Contact info (combine all contacts from properties and inquiries)
        all_contacts = []
        for prop in self.properties:
            all_contacts.extend(prop.contact)
        for inq in self.inquiries:
            all_contacts.extend(inq.contact_info)
        contact_info = {
            "contact_numbers": list(set(all_contacts))
        }

        # Structured content with indexed property listings and inquiries
        structured_content = {}
        
        # Add property listings
        for i, prop in enumerate(self.properties, 1):
            structured_content[f"property_listing_{i}"] = prop.to_dict()
            
        # Add inquiries
        for i, inq in enumerate(self.inquiries, 1):
            structured_content[f"real_estate_inquiry_{i}"] = inq.to_dict()

        return {
            f"message_no_{self.message_index:03d}_structured_format": {
                "basic_message_info": basic_info,
                "contact_info": contact_info,
                "structured_message_content": structured_content,
                "raw_text": self.raw_text
            }
        }

    def _determine_message_type(self) -> str:
        if self.properties and self.inquiries:
            return "mixed"
        elif self.properties:
            return "property_listing"
        elif self.inquiries:
            return "inquiry"
        return "other"

class MistralExtractor:
    def __init__(self, api_key: str, batch_size: int = 10, max_retries: int = 3):
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.api_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.cache = {}  # Simple in-memory cache
        
        # Property boundary markers
        self.PROPERTY_MARKERS = {
            'strong': [
                r"\*{1,2}[^*]+\*{1,2}",  # Text between asterisks
                r"(?:\n|\r\n){2,}",       # Double line breaks
                r"(?:\d+\s*(?:bhk|rk))",  # BHK/RK configurations
                r"(?:commercial|residential|office|shop)\s+(?:space|property|available)",
            ],
            'weak': [
                r"(?:also|another|available|more)",
                r"(?:contact|call).*?:\s*[\d\s-]+",
                r"(?:rent|deposit|price).*?(?:\d+(?:k|l|cr)?)",
            ]
        }

    @lru_cache(maxsize=1000)
    def _get_cached_property_type(self, text: str) -> str:
        """Cache property type detection to avoid redundant processing"""
        if re.search(r'\b(?:bhk|rk|flat|apartment)\b', text, re.I):
            return "residential"
        elif re.search(r'\b(?:office|shop|commercial)\b', text, re.I):
            return "commercial"
        elif re.search(r'\b(?:plot|land|acre)\b', text, re.I):
            return "land"
        return "unknown"

    def _identify_property_boundaries(self, text: str) -> List[int]:
        """Identify potential property boundaries using markers"""
        boundaries = set()
        
        # Find all strong markers
        for pattern in self.PROPERTY_MARKERS['strong']:
            for match in re.finditer(pattern, text, re.I):
                boundaries.add(match.start())
                
        # Only add weak markers if they're near other markers or have supporting context
        for pattern in self.PROPERTY_MARKERS['weak']:
            for match in re.finditer(pattern, text, re.I):
                pos = match.start()
                context = text[max(0, pos-50):min(len(text), pos+50)]
                
                # Check if there's supporting evidence
                if any(re.search(strong, context, re.I) for strong in self.PROPERTY_MARKERS['strong']):
                    boundaries.add(pos)
        
        return sorted(list(boundaries))

    async def _split_with_outlines(self, text: str) -> List[str]:
        """Use Mistral API for intelligent property splitting"""
        prompt = f"""Split this real estate message into individual property listings.
        Each property should be marked with '###PROPERTY###'.
        
        Input: {text}
        
        Guidelines:
        1. Look for these property separators:
           - Asterisk-marked sections (*Property*)
           - Property type mentions (BHK, Shop, Office)
           - Location changes
           - Price/rent mentions
           - Contact number blocks
        2. Each property should have:
           - Type (residential/commercial/land)
           - Location
           - Price information
           - Contact details
        3. Keep related information together
        4. Don't split mid-property
        
        Format:
        ###PROPERTY###
        [Complete property 1 details]
        ###PROPERTY###
        [Complete property 2 details]
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                result = await self._make_api_call(session, [text])
                if result and isinstance(result, list):
                    properties = [p.strip() for p in result if p.strip()]
                    return properties if properties else [text]
        except Exception as e:
            logger.warning(f"Property splitting failed: {e}")
        return [text]

    def _sliding_window_analysis(self, text: str, window_size: int = 150, overlap: int = 50) -> List[Dict[str, Any]]:
        """Advanced sliding window analysis for property boundaries"""
        words = text.split()
        windows = []
        
        for i in range(0, len(words), window_size - overlap):
            window = words[i:i + window_size]
            window_text = " ".join(window)
            
            # Analyze window content
            analysis = {
                'start': i,
                'text': window_text,
                'features': {
                    'has_property_type': bool(re.search(r'\b(?:bhk|office|shop|plot)\b', window_text, re.I)),
                    'has_price': bool(re.search(r'\b(?:rent|price|deposit)\b.*?\d+', window_text, re.I)),
                    'has_contact': bool(re.search(r'(?:contact|call).*?\d{10}', window_text, re.I)),
                    'has_location': bool(re.search(r'\b(?:in|at|near|locality|area)\b', window_text, re.I)),
                }
            }
            
            # Score the window for property completeness
            analysis['completeness_score'] = sum(analysis['features'].values())
            windows.append(analysis)
            
        return windows

    def _is_boundary(self, current_text: str, previous_text: str) -> bool:
        """Check if there's a property boundary between two text segments"""
        # Check for strong markers at the start of current text
        for pattern in self.PROPERTY_MARKERS['strong']:
            if re.match(pattern, current_text, re.I):
                return True
        
        # Check for context switches
        context_switches = [
            (r'\b(?:another|also|different)\b.*?\b(?:property|flat|office|shop)\b', 0.8),
            (r'\b(?:contact|call)\s*(?:[\d\s-]+|[^.]*?\d{10})', 0.6),
            (r'\b(?:available|located)\s+(?:in|at|near)\b', 0.7),
        ]
        
        score = 0
        for pattern, weight in context_switches:
            if re.search(pattern, current_text, re.I):
                score += weight
        
        return score >= 0.8  # Return True if strong evidence of boundary

    def _merge_property_candidates(self, windows: List[Dict[str, Any]]) -> List[str]:
        """Merge window analysis into coherent property listings"""
        properties = []
        current_property = []
        current_score = 0
        
        for i, window in enumerate(windows):
            # Start new property if:
            # 1. Current window has high completeness and previous property is substantial
            # 2. Clear boundary marker is found
            # 3. Context switch detected
            
            if (window['completeness_score'] >= 3 and current_score > 0) or \
               (i > 0 and self._is_boundary(window['text'], windows[i-1]['text'])):
                
                if current_property:
                    properties.append(" ".join(current_property))
                current_property = [window['text']]
                current_score = window['completeness_score']
            else:
                current_property.append(window['text'])
                current_score = max(current_score, window['completeness_score'])
        
        if current_property:
            properties.append(" ".join(current_property))
            
        return properties

    async def _split_multiple_properties(self, text: str) -> List[str]:
        """Main method to split text into multiple property listings"""
        # Try Mistral API first
        outline_properties = await self._split_with_outlines(text)
        
        # If outlines found multiple properties, validate and return
        if len(outline_properties) > 1:
            validated_properties = [p for p in outline_properties if self._is_valid_property(p)]
            if validated_properties:
                return validated_properties
        
        # Fall back to sliding window analysis
        windows = self._sliding_window_analysis(text)
        property_candidates = self._merge_property_candidates(windows)
        
        # Validate final properties
        final_properties = [p for p in property_candidates if self._is_valid_property(p)]
        
        return final_properties if final_properties else [text]

    def _is_valid_property(self, text: str) -> bool:
        """Validate if text contains real estate related information"""
        
        # Skip messages that are clearly not property related
        skip_patterns = [
            r'<Media omitted>',  # Skip media messages
            r'This message was deleted',  # Skip deleted messages
            r'^\s*$'  # Skip empty messages
        ]
        
        if any(re.search(pattern, text, re.I) for pattern in skip_patterns):
            return False
        
        # Check for ANY real estate related indicators
        property_indicators = [
            # Property Types
            r'\b(?:bhk|rk|flat|apartment|house|villa|property|space|office|shop|showroom|plot|land)\b',
            r'\b(?:commercial|residential|rental|rent|sale|lease)\b',
            r'\b(?:room|building|complex|tower|floor|storey)\b',
            
            # Transaction Types
            r'\b(?:available|required|looking|wanted|need|urgent|sale|rent|lease)\b',
            
            # Property Features
            r'\b(?:furnished|unfurnished|semi|carpet|built|sq\.?\s*ft|sqft|square\s*feet)\b',
            r'\b(?:bedroom|bath|parking|garden|terrace|balcony)\b',
            
            # Price Indicators
            r'\b(?:\d+(?:\.\d+)?(?:\s*(?:k|l|lac|lakh|cr|crore))?)\b',
            r'(?:rs\.?|inr|₹|price|budget|rent|deposit)',
            r'@\s*\d+',  # Matches patterns like "@ 1.25"
            
            # Location Indicators
            r'\b(?:location|area|society|complex|road|street|lane|near|opposite|beside)\b',
            r'\b(?:west|east|north|south)\b',
            
            # Contact/Transaction
            r'\b(?:contact|call|whatsapp|owner|broker|agent|inspection|site\s*visit)\b',
            r'\b\d{10}\b',  # Phone numbers
            
            # Requirements/Inquiries
            r'\b(?:requirement|looking|wanted|need|searching|urgent)\b'
        ]
        
        # Count how many indicators are present
        matches = sum(1 for pattern in property_indicators if re.search(pattern, text, re.I))
        
        # If we find at least 2 different indicators, consider it valid
        return matches >= 2

    async def _process_property(self, text: str, session: aiohttp.ClientSession) -> Optional[List[Dict]]:
        """Process a single message that may contain multiple properties"""
        for retry in range(self.max_retries):
            try:
                # Extract property information
                results = await self._make_api_call(session, [text])
                if not results:
                    continue
                
                # Process each property in the results
                cleaned_results = []
                for result in results:
                    cleaned_result = self._validate_property_data(result, text)
                    if cleaned_result:
                        cleaned_results.append(cleaned_result)
                
                if cleaned_results:
                    return cleaned_results
                    
            except Exception as e:
                logger.warning(f"Property processing attempt {retry + 1} failed: {e}")
                await asyncio.sleep(2 ** retry)  # Exponential backoff
                
        return None

    def _validate_property_data(self, data: Dict, original_text: str) -> Optional[Dict]:
        """Validate and clean extracted property data"""
        try:
            # Ensure required fields are present
            required_fields = {
                'property_category',
                'location',
                'area',
                'price'
            }
            
            if not all(field in data for field in required_fields):
                return None
                
            # Clean and validate specific fields
            if 'price' in data:
                data['price'] = self._clean_price_data(data['price'])
            if 'area' in data:
                data['area'] = self._clean_area_data(data['area'])
            if 'contact' in data:
                data['contact'] = self._extract_contact_numbers(original_text)
                
            return data
                
        except Exception as e:
            logger.warning(f"Property validation failed: {e}")
            return None

    async def _process_message(self, message_data: Tuple[str, str, str], message_index: int) -> WhatsAppMessage:
        """Process a single message with multiple properties"""
        timestamp, sender, content = message_data
        
        # Create message container with index
        message = WhatsAppMessage(
            timestamp=timestamp,
            sender=sender,
            raw_text=content,
            message_index=message_index
        )
        
        # Split into multiple properties
        property_texts = await self._split_multiple_properties(content)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prop_text in property_texts:
                task = self._process_property(prop_text, session)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Create property objects for valid results
            for result_list in results:
                if result_list:
                    # Handle multiple properties from a single result
                    for result in result_list:
                        prop_obj = self._create_property_object(result)
                        if prop_obj:
                            message.properties.append(prop_obj)
        
        return message

    async def process_chat_async(self, chat_text: str) -> List[Dict]:
        message_pattern = re.compile(
            r'(\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*?)(?=(?:\d{1,2}/\d{1,2}/\d{4}|$))',
            re.DOTALL
        )
        
        messages = list(message_pattern.finditer(chat_text))
        logger.info(f"Found {len(messages)} total messages in chat")
        
        processed_messages = []
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = min(5, len(messages))  # Process 5 messages at a time
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                logger.info(f"\nProcessing batch {i//batch_size + 1} with {len(batch)} messages")
                
                # Pre-process messages to identify property types
                batch_texts = []
                batch_indices = []
                for idx, match in enumerate(batch):
                    timestamp, sender, content = match.groups()
                    logger.info(f"\nAnalyzing message {i + idx}:")
                    logger.info(f"Content: {content[:200]}...")
                    
                    if self._is_valid_property(content):
                        logger.info(f"✓ Found valid property in message {i + idx}")
                        batch_texts.append(content)
                        batch_indices.append(idx)
                    else:
                        logger.info(f"✗ Message {i + idx} did not contain valid property")
                
                if batch_texts:
                    logger.info(f"\nMaking API call for {len(batch_texts)} valid properties")
                    results = await self._make_api_call(session, batch_texts)
                    
                    if results:
                        logger.info(f"Received {len(results)} results from API")
                        # Process each message and its properties
                        for idx, (result, batch_idx) in enumerate(zip(results, batch_indices)):
                            if result is None:
                                logger.info(f"Result {idx} was None")
                                continue
                                
                            match = batch[batch_idx]
                            timestamp, sender, content = match.groups()
                            message = WhatsAppMessage(
                                timestamp=timestamp,
                                sender=sender,
                                raw_text=content,
                                message_index=i + batch_idx
                            )
                            
                            # Process all properties for this message
                            properties_list = result  # result is already a list of properties
                            for prop_data in properties_list:
                                if prop_data:  # Skip None values
                                    prop_obj = self._create_property_object(prop_data)
                                    if prop_obj:
                                        message.properties.append(prop_obj)
                            
                            # Only add message if it has properties
                            if message.properties:
                                logger.info(f"Adding message {i + batch_idx} with {len(message.properties)} properties")
                                processed_messages.append(message.to_dict())
                
                # Progress indication
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(messages)-1)//batch_size + 1}")
                logger.info(f"Found {len(processed_messages)} properties so far")
                
                # Add a small delay between batches to avoid rate limiting
                await asyncio.sleep(1)
        
        logger.info(f"Final count of processed messages: {len(processed_messages)}")
        return processed_messages

    def _get_extraction_prompt(self, text: str) -> str:
        """Enhanced prompt with examples for better extraction"""
        return f"""Extract real estate information from WhatsApp messages. First determine the property category, then extract details according to the schema.

Here are examples of different property types and their correct extractions:

Example 1 (Residential):
Message: "2BHK apartment available in Andheri West, 800 sq ft carpet area. Rent 45000, deposit 1.5L. 3rd floor with lift. Semi furnished with modular kitchen. Family preferred. Contact: 9876543210"
Output:
{{
    "property_category": "residential",
    "configuration": "2BHK",
    "location": "Andheri West",
    "area": {{
        "value": "800",
        "unit": "carpet"
    }},
    "price": {{
        "rent": "45000",
        "deposit": "150000"
    }},
    "floor": "3rd",
    "total_floors": "",
    "furnishing_status": "Semi furnished",
    "amenities": ["modular kitchen", "lift"],
    "preferred_tenants": ["Family"],
    "parking": {{
        "two_wheeler": 0,
        "four_wheeler": 0
    }},
    "additional_details": [],
    "contact": ["9876543210"]
}}

Example 2 (Commercial):
Message: "Shop available for rent on main road Borivali West. 500 sq ft carpet, ground floor, road facing. Water connection available. Rent 80k, deposit 3L. Suitable for restaurant/retail. Call 8765432109"
Output:
{{
    "property_category": "commercial",
    "property_type": "shop",
    "location": "Borivali West",
    "area": {{
        "value": "500",
        "unit": "carpet"
    }},
    "price": {{
        "rent": "80000",
        "deposit": "300000"
    }},
    "floor": "ground floor",
    "furnishing_status": "",
    "washroom": false,
    "parking_available": false,
    "road_facing": true,
    "water_connection": true,
    "suitable_for": ["restaurant", "retail"],
    "additional_details": ["main road"],
    "contact": ["8765432109"]
}}

Example 3 (Land):
Message: "Industrial plot available near MIDC. 1000 sq yards, 40ft road width. All boundaries done. Corner plot with development potential. Price negotiable. Contact Mr. Sharma: 7654321098"
Output:
{{
    "property_category": "land",
    "plot_type": "Industrial",
    "location": "near MIDC",
    "area": {{
        "value": "1000",
        "unit": "sq yards"
    }},
    "price": {{
        "rent": "",
        "deposit": ""
    }},
    "road_width": "40ft",
    "boundaries": true,
    "corner_plot": true,
    "legal_restrictions": [],
    "development_potential": ["development potential"],
    "additional_details": ["Price negotiable"],
    "contact": ["7654321098"]
}}

Example 4 (Multiple Properties in One Message):
Message: "*Office Space* Andheri East, 1000 sqft, fully furnished, 1L rent. *Also Available* 2BHK flat same building, 800 sqft, semi furnished, 50k rent. Contact: 9876543210"
Output: [
    {{
        "property_category": "commercial",
        "property_type": "office",
        "location": "Andheri East",
        "area": {{
            "value": "1000",
            "unit": "sqft"
        }},
        "price": {{
            "rent": "100000",
            "deposit": ""
        }},
        "furnishing_status": "fully furnished",
        "additional_details": [],
        "contact": ["9876543210"]
    }},
    {{
        "property_category": "residential",
        "configuration": "2BHK",
        "location": "Andheri East",
        "area": {{
            "value": "800",
            "unit": "sqft"
        }},
        "price": {{
            "rent": "50000",
            "deposit": ""
        }},
        "furnishing_status": "semi furnished",
        "additional_details": ["same building"],
        "contact": ["9876543210"]
    }}
]

Now extract information from this message:
{text}

Important rules:
1. ANY mention of property (flat, room, shop, office, etc.) should be considered valid
2. Prices can be in various formats (25000/-, 1.25cr, 80k, etc.)
3. Location can be partial (just area name or direction like 'west')
4. For multiple properties, return an array of objects
5. If exact values aren't given, use empty strings
6. Include all contact numbers found
7. Consider 'rental', 'for rent', 'available' as indicators of rental properties
8. Extract all numbers that could be prices or areas
9. Mark boolean fields as true only if explicitly mentioned
10. When in doubt about category:
    - If mentions BHK/RK/flat/apartment -> residential
    - If mentions shop/office/commercial -> commercial
    - If mentions plot/land -> land
    - Default to residential for general rooms/spaces

Return the JSON in the exact schema format shown in the examples."""

    async def _make_api_call(self, session: aiohttp.ClientSession, texts: List[str], retry_count: int = 0) -> Optional[List[Dict]]:
        """Make a batch API call to Mistral"""
        try:
            # Process each text separately to maintain message boundaries
            results = []
            for text in texts:
                async with session.post(
                    self.api_endpoint,
                    json={
                        "model": "mistral-medium",
                        "messages": [{
                            "role": "user",
                            "content": self._get_extraction_prompt(text)
                        }],
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "top_p": 0.95,
                        "response_format": {"type": "json_object"}
                    },
                    headers=self.headers,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        try:
                            data = json.loads(content)
                            # If we got a list of properties, keep it as is
                            # If we got a single property, wrap it in a list
                            results.append(data if isinstance(data, list) else [data])
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {str(e)}")
                            results.append(None)
                    else:
                        results.append(None)
                # Add a small delay between API calls
                await asyncio.sleep(1)
            return results
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_api_call(session, texts, retry_count + 1)
            return None

    def _create_property_object(self, data: Dict) -> Union[ResidentialProperty, CommercialProperty, LandProperty]:
        """Create appropriate property object based on category"""
        category = data.get("property_category", "").lower()
        
        if category == "residential":
            return ResidentialProperty(
                property_category=PropertyCategory.RESIDENTIAL,
                configuration=data.get("configuration", ""),
                location=data.get("location", ""),
                area=data.get("area", {}),
                price=data.get("price", {}),
                floor=data.get("floor", ""),
                total_floors=data.get("total_floors", ""),
                furnishing_status=data.get("furnishing_status", ""),
                amenities=data.get("amenities", []),
                preferred_tenants=data.get("preferred_tenants", []),
                parking=data.get("parking", {"two_wheeler": 0, "four_wheeler": 0}),
                additional_details=data.get("additional_details", []),
                contact=data.get("contact", []),
                raw_text=data.get("raw_text", "")
            )
        elif category == "commercial":
            return CommercialProperty(
                property_category=PropertyCategory.COMMERCIAL,
                property_type=data.get("property_type", ""),
                location=data.get("location", ""),
                area=data.get("area", {}),
                price=data.get("price", {}),
                floor=data.get("floor", ""),
                furnishing_status=data.get("furnishing_status", ""),
                washroom=data.get("washroom", False),
                parking_available=data.get("parking_available", False),
                road_facing=data.get("road_facing", False),
                water_connection=data.get("water_connection", False),
                suitable_for=data.get("suitable_for", []),
                additional_details=data.get("additional_details", []),
                contact=data.get("contact", []),
                raw_text=data.get("raw_text", "")
            )
        else:  # land
            return LandProperty(
                property_category=PropertyCategory.LAND,
                plot_type=data.get("plot_type", ""),
                location=data.get("location", ""),
                area=data.get("area", {}),
                price=data.get("price", {}),
                road_width=data.get("road_width", ""),
                boundaries=data.get("boundaries", False),
                corner_plot=data.get("corner_plot", False),
                legal_restrictions=data.get("legal_restrictions", []),
                development_potential=data.get("development_potential", []),
                additional_details=data.get("additional_details", []),
                contact=data.get("contact", []),
                raw_text=data.get("raw_text", "")
            )

    def _clean_price_data(self, price_data: Dict[str, str]) -> Dict[str, str]:
        """Clean and standardize price data"""
        def convert_to_number(value: str) -> str:
            if not value:
                return ""
            # Remove non-numeric characters except dots
            num = re.sub(r'[^\d.]', '', value)
            # Convert lakhs/crores to actual numbers
            if 'cr' in value.lower():
                num = str(float(num) * 10000000)
            elif 'l' in value.lower() or 'lac' in value.lower():
                num = str(float(num) * 100000)
            elif 'k' in value.lower():
                num = str(float(num) * 1000)
            return num

        return {
            'rent': convert_to_number(price_data.get('rent', '')),
            'deposit': convert_to_number(price_data.get('deposit', ''))
        }

    def _clean_area_data(self, area_data: Dict[str, str]) -> Dict[str, str]:
        """Clean and standardize area data"""
        return {
            'value': re.sub(r'[^\d.]', '', area_data.get('value', '')),
            'unit': area_data.get('unit', '').lower().strip()
        }

    def _extract_contact_numbers(self, text: str) -> List[str]:
        """Extract contact numbers from text"""
        numbers = re.findall(r'\b\d{10}\b', text)
        return list(set(numbers))  # Remove duplicates

    async def process_file(self, filepath: str, output_path: str, limit: Optional[int] = None):
        """Process a chat file with parallel processing"""
        try:
            # First, check if the file exists
            logger.info(f"Attempting to read file: {filepath}")
            
            # Read the entire file content
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                chat_text = await f.read()
            
            logger.info(f"Successfully read file, size: {len(chat_text)} bytes")
            
            # Find all message timestamps
            timestamp_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}')
            timestamps = list(timestamp_pattern.finditer(chat_text))
            logger.info(f"Found {len(timestamps)} messages in file")
            
            if limit and len(timestamps) > limit:
                logger.info(f"Limiting to first {limit} messages")
                # Find the position of the last message we want to process
                last_message_pos = timestamps[limit].start()
                chat_text = chat_text[:last_message_pos]
                logger.info(f"Truncated text to {len(chat_text)} bytes")
            
            # Print first few characters of the text to verify content
            logger.info(f"First 200 characters of chat text: {chat_text[:200]}")
            
            results = await self.process_chat_async(chat_text)
            logger.info(f"Processed {len(results)} messages")
            
            # Save results
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, ensure_ascii=False))
            
            return results
        
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

def main():
    try:
        print("\n🚀 Starting real estate extraction process...")
        
        parser = argparse.ArgumentParser(description='Process WhatsApp chat messages for real estate data')
        parser.add_argument('--limit', type=int, default=10, help='Number of messages to process (default: 10)')
        parser.add_argument('--input', type=str, default='text/chats.txt', help='Input file path')
        parser.add_argument('--output', type=str, default='output40.json', help='Output file path')
        parser.add_argument('--batch-size', type=int, default=5, help='Batch size for parallel processing')
        
        args, unknown = parser.parse_known_args()
        
        print("\n⚙️ Configuration:")
        print(f"  Input file: {args.input}")
        print(f"  Output file: {args.output}")
        print(f"  Batch size: {args.batch_size}")
        print("  Model: mistral-medium")
        print(f"  Message limit: {args.limit}")
        print("\n")
        
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"\n❌ Error: Input file '{args.input}' does not exist!")
            return
        
        extractor = MistralExtractor(
            api_key="uFlkW5B5rarhRCcnj7UMDnP2itksQYJb",
            batch_size=args.batch_size
        )
        
        results = asyncio.run(extractor.process_file(args.input, args.output, args.limit))
        print(f"\n✅ Processing complete! Found {len(results)} messages with properties")
        print(f"Results saved to {args.output}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()