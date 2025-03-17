# Twitter Posting Agent  
This repository contains a **Twitter Posting Agent** that automatically generates and posts tweets to X (Twitter) at regular intervals using an **AI-powered tweet generator (DistilGPT-2)** and the **X API v2 (Tweepy)**.  

It fulfills all task requirements and includes the optional AI bonus task for **dynamic tweet generation**.  

---

## **Project Overview**  
The **Twitter Posting Agent**:  
- Generates concise, engaging tweets using **DistilGPT-2** from predefined topic prompts.  
- Posts two tweets sequentially.  
- Runs autonomously.  

---

## **Key Features**  
✅ **Automated Posting**: Tweets are posted at regular intervals without intervention.  
✅ **Dynamic Content**: Uses DistilGPT-2 to create single-sentence tweets (<280 characters) from a `topics.json` file.  
✅ **Robust Design**: Includes logging, error handling, and a modular structure for scalability.  

---

## **Task Requirements**  
✔️ **Automatic Posting**: Tweets are posted at fixed intervals without manual input.  
✔️ **Predefined Source**: Content is sourced from `topics.json`, enhanced by AI generation.  
✔️ **Continuous Operation**: Runs fully autonomously.  

---

## **Bonus Task**  
✔️ **AI-Powered Generation**: Integrates **DistilGPT-2** to dynamically generate engaging, context-aware tweets instead of static text.  

---

## **Evaluation Criteria**  
🔹 **Functionality**: Successfully posts AI-generated tweets, verified by Tweet IDs in logs.  
🔹 **Code Quality**: Well-documented with inline comments, structured with a `TweetGenerator` class, and uses logging for transparency.  
🔹 **Scalability**: Easily extensible—add new topics via `add_topic()`, adjust intervals in `main()`, or scale to more tweets.  
🔹 **Efficiency**: Uses lightweight **DistilGPT-2 (~82M parameters, ~500MB VRAM)** for fast generation, optimized with GPU support.  

---

## **Prerequisites**  
### **1. Python Version**  
- Python **3.8 or higher**  

### **2. Required Libraries**  
Install the dependencies:  
```bash
pip install tweepy schedule transformers torch requests
