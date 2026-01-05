import nltk
import os

try:
    print("Checking NLTK data...")
    # Add project directory to nltk path
    nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
    
    # Check/Download punkt
    try:
        nltk.data.find('tokenizers/punkt')
        print("punkt already available")
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt', quiet=False)
    
    print("NLTK data check complete")

except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    # Don't fail the build, just warn - app might still work or try again
    pass
