# 🧬 Pharmaceutical Description Generator

AI-powered pharmaceutical product description generator using cloud-based LLMs for compliance-focused, professional descriptions.

---

## 🚀 **Quick Start**

### **1. Installation**

```powershell
# Clone or download the repository
cd C:\Users\manis\Downloads\C4\PharmaDescription-Generator\pharma-description-generator

# Install dependencies
pip install -r requirements.txt
```

### **2. Run the Application**

```powershell
python app.py
```

### **3. Access the Web Interface**

Open your browser to: **http://127.0.0.1:5000**

---

## 📋 **Features**

✅ **Cloud-Based LLM Models**: Mistral 7B, Gemini 1.5 Flash, OpenChat 7B, DeepSeek R1, GPT-OSS 20B  
✅ **10 Concurrent Requests**: High-speed batch processing  
✅ **Zero-Fail Fallback**: Automatic Excel-based descriptions if AI fails  
✅ **Pharmaceutical Compliance**: No medical claims, mandatory disclaimers  
✅ **Strict Templates**: 4 bullets for short, 7-8 sentences for long descriptions  
✅ **Rich Data Extraction**: Uses ALL Excel columns (ingredients, benefits, directions, safety, manufacturer, strength, form)  
✅ **Progress Tracking**: Real-time status updates and stop functionality  

---

## 🤖 **Supported Models**

| Model | Provider | Speed | Quality | Cost |
|-------|----------|-------|---------|------|
| **Mistral 7B** | OpenRouter | ⚡⚡⚡ | ⭐⭐⭐⭐ | $ |
| **Gemini 1.5 Flash** | Google | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | $ |
| **OpenChat 7B** | OpenRouter | ⚡⚡ | ⭐⭐⭐ | $ |
| **DeepSeek R1** | OpenRouter | ⚡⚡ | ⭐⭐⭐⭐ | $ |
| **GPT-OSS 20B** | OpenRouter | ⚡ | ⭐⭐⭐⭐⭐ | $$ |

---

## 🔑 **API Key Setup**

### **Option 1: Gemini (Recommended)**
1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Paste into the web interface

### **Option 2: OpenRouter (Multiple Models)**
1. Visit: https://openrouter.ai/keys
2. Sign up and create an API key
3. Add credits to your account
4. Use with Mistral, OpenChat, DeepSeek, or GPT-OSS

---

## 📁 **Excel File Format**

Your input Excel file should have the following structure:

| Product Name | Category | Ingredients | Benefits | Directions | Safety | Manufacturer | Strength | Form |
|--------------|----------|-------------|----------|------------|--------|--------------|----------|------|
| Product 1 | Health Condition > Bleeding Disorders | Active ingredients... | Intended for... | Take as directed... | Keep out of reach... | Company Name | 500mg | Tablet |

**Required Column:**
- First column: Product Name (mandatory)

**Optional Columns** (extracted automatically):
- Category, Ingredients, Benefits, Directions, Safety, Manufacturer, Strength, Form, Description

---

## ⚙️ **Configuration**

### **Batch Processing Settings**
- **Concurrent Requests**: 10 (optimized for cloud APIs)
- **Timeout per Product**: 60 seconds
- **Timeout per Batch**: 300 seconds
- **Rate Limiting**: Adaptive delays (0.1-5.0 seconds)

### **Description Requirements**
- **Short Description**: Exactly 4 HTML bullet points (`<ul><li>...</li></ul>`)
- **Long Description**: Exactly 7-8 sentences
- **Mandatory Disclaimer**: Auto-appended to all long descriptions
- **No Medical Claims**: Automatic filtering of forbidden words (cure, treat, heal, etc.)

---

## 🛡️ **Pharmaceutical Compliance**

### **Forbidden Words Filter**
The system automatically replaces medical claim words:
- cure → support
- treat → intended for
- heal → support
- guarantee → formulated to
- diagnose → identify
- prevent → may support
- therapy → routine
- therapeutic → beneficial

### **Mandatory Disclaimer**
All long descriptions include:
> **Important Note: This information is for general product purposes only and is not intended as medical advice. Always consult a healthcare professional before use.**

---

## 📊 **Processing Capacity**

- **Maximum File Size**: 50 MB
- **Maximum Products**: 50,000 per file
- **Recommended Batch Size**: 100-1,000 products

---

## 🔧 **Dependencies**

```
Flask >= 3.0.0          # Web framework
pandas >= 2.0.0         # Excel processing
openpyxl >= 3.1.0       # Excel file handling
Werkzeug >= 3.0.0       # WSGI utilities
google-generativeai >= 0.8.0  # Gemini API
httpx >= 0.28.0         # Async HTTP client
```

---

## 📝 **Output Format**

The generated Excel file includes:
- **Product Name** (original)
- **Short Description** (4 HTML bullets)
- **Long Description** (7-8 sentences with disclaimer)
- **All Original Columns** (preserved from input)

---

## 🚨 **Troubleshooting**

### **Issue: "API Key Invalid"**
- Verify your API key is correct
- Check API credits/quota
- Test the key using the "Test API" button

### **Issue: "Processing Too Slow"**
- Reduce batch size (fewer concurrent requests)
- Use faster models (Gemini 1.5 Flash, Mistral 7B)
- Check internet connection speed

### **Issue: "Empty Descriptions"**
- Check if Excel file has data in first column
- Verify API key has available credits
- Zero-fail fallback will generate basic descriptions from Excel data

### **Issue: "Stop Button Not Working"**
- Processing stops after current batch completes
- Partial results are saved automatically
- Check job status in progress panel

---

## 🎯 **Best Practices**

1. **Start Small**: Test with 5-10 products first
2. **Use Rich Data**: Include ingredients, benefits, directions in Excel
3. **Monitor Progress**: Watch real-time updates in web interface
4. **Check Output**: Review generated descriptions for quality
5. **API Credits**: Monitor your API usage to avoid interruptions

---

## 📞 **Support**

For issues or questions:
1. Check `pharma_generator.log` for error details
2. Run `python diagnose.py` to check system health
3. Review the error message in the web interface

---

## 📄 **License**

Proprietary - For internal use only

---

## 🎉 **Version**

**Current Version**: 2.0.0 (Cloud-Only Edition)

**Last Updated**: December 13, 2025

**Architecture**: Flask + Async Cloud APIs + Zero-Fail Fallback
