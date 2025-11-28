import json
from pathlib import Path

# Create a small demo dataset for deployment
demo_chunks = [
    {
        "text": "Diabetes symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision. Type 2 diabetes often develops gradually.",
        "specialty": "Endocrinology", 
        "description": "Diabetes mellitus symptoms and diagnosis",
        "source_id": 1
    },
    {
        "text": "Hypertension treatment typically involves lifestyle changes and medications. ACE inhibitors and diuretics are commonly prescribed.",
        "specialty": "Cardiology",
        "description": "Hypertension management and treatment options", 
        "source_id": 2
    },
    {
        "text": "Heart attack symptoms include chest pain, shortness of breath, nausea, and pain radiating to the arm or jaw. Seek immediate medical attention.",
        "specialty": "Cardiology",
        "description": "Myocardial infarction signs and symptoms",
        "source_id": 3
    }
]

# Save demo data
demo_dir = Path(__file__).parent.parent / "demo_data"
demo_dir.mkdir(exist_ok=True)

with open(demo_dir / "demo_chunks.json", 'w') as f:
    json.dump(demo_chunks, f, indent=2)

print("Demo dataset created!")