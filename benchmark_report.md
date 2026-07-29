# STT Benchmark Report — Google Cloud STT v2 (Production Config)

Generated from a TTS -> STT round-trip test: each of the 60 dataset questions was synthesized to speech with Google Cloud TTS (en-IN Chirp3-HD voice) and the resulting audio was fed through the exact same `transcribe()` path used in production (`companies/core/stt_google.py`, STT v2, `latest_long` model, `global` location, `language_codes=[en-IN, hi-IN, te-IN]`).

## Important Caveats — Read Before Trusting These Numbers

1. **No live caller was used.** There is no recording capability in this session, so real human speech was not available. This measures STT's raw recognition ability on clean, noise-free, single-speaker synthetic audio — real phone calls have background noise, telephone compression, natural disfluencies, and genuine speaker accents, all of which this test cannot capture. Expect real accuracy to be worse.
2. **All 60 questions were synthesized with the same en-IN English voice**, including the 25 Hindi and 25 Telugu (Romanized) questions — there is no native Hindi/Telugu script text in this dataset (by design) to feed a hi-IN/te-IN voice. The en-IN voice frequently mispronounces short/common words in its own stylized way (observed: "to" -> "Tu", "the" -> "D", "are" -> "R") — some WER/CER penalty below reflects **TTS mispronunciation**, not an STT failure, especially visible in the English results.
3. **Language-detection accuracy is not meaningful here** — since every clip was voiced in English phonetics, `en-IN` is the phonetically correct detection for all 60 rows regardless of the question's labeled language. This benchmark cannot validate real multilingual auto-detection; it only confirms the recognizer didn't spuriously flip to hi-IN/te-IN on English-voiced audio.
4. **Pass/Fail threshold:** WER <= 0.35 on normalized (lowercased, punctuation-stripped) text. This is a lenient bar chosen for phone-quality code-switched speech; tune once real-call data exists.

---

## Overall Results

- **Overall Accuracy (Pass rate):** 36.7% (22/60)
- **Failure Rate:** 63.3% (38/60)
- **Average WER:** 0.4758
- **Average CER:** 0.224
- **Average Similarity Score:** 0.8395
- **Average Recognition Time:** 1.784s
- **Median Recognition Time:** 1.61s
- **Average Confidence:** 0.5737
- **"en-IN" Detection Rate:** 100.0% (expected under this all-English-voice setup — see caveat 3)

## Accuracy Per Language

| Language | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) | Avg Confidence |
|---|---|---|---|---|---|---|---|---|
| English | 10 | 1 | 9 | 10.0% | 0.4755 | 0.2991 | 2.11 | 0.6549 |
| Hindi | 25 | 21 | 4 | 84.0% | 0.206 | 0.1073 | 1.496 | 0.7682 |
| Telugu | 25 | 0 | 25 | 0.0% | 0.7457 | 0.3106 | 1.942 | 0.3372 |

## Accuracy Per Category

| Category | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Addresses | 3 | 1 | 2 | 33.3% | 0.3547 | 0.2203 | 2.123 |
| Business Hours | 3 | 1 | 2 | 33.3% | 0.3651 | 0.2094 | 1.332 |
| Complaint Handling | 5 | 2 | 3 | 40.0% | 0.452 | 0.2419 | 2.627 |
| Contact Information | 2 | 1 | 1 | 50.0% | 0.2 | 0.06 | 1.001 |
| Customer Support | 2 | 1 | 1 | 50.0% | 0.4 | 0.0781 | 0.992 |
| Dates | 2 | 1 | 1 | 50.0% | 0.8428 | 0.4906 | 1.175 |
| Delivery | 5 | 1 | 4 | 20.0% | 0.5176 | 0.1834 | 1.707 |
| FAQ | 2 | 1 | 1 | 50.0% | 0.3555 | 0.2561 | 1.23 |
| Follow-up Questions | 2 | 1 | 1 | 50.0% | 0.4924 | 0.17 | 2.208 |
| General Conversation | 4 | 2 | 2 | 50.0% | 0.4813 | 0.2278 | 1.099 |
| Greetings | 3 | 2 | 1 | 66.7% | 0.2917 | 0.0766 | 1.675 |
| Mixed English+Hindi | 2 | 2 | 0 | 100.0% | 0.0 | 0.0 | 1.753 |
| Mixed English+Telugu | 2 | 0 | 2 | 0.0% | 0.7 | 0.4415 | 2.776 |
| Numbers | 3 | 0 | 3 | 0.0% | 0.4962 | 0.1988 | 2.196 |
| Order Status | 5 | 2 | 3 | 40.0% | 0.55 | 0.2562 | 2.101 |
| Phone Numbers | 3 | 0 | 3 | 0.0% | 0.8619 | 0.5725 | 2.272 |
| Pricing | 5 | 1 | 4 | 20.0% | 0.5576 | 0.2282 | 1.63 |
| Product Information | 5 | 2 | 3 | 40.0% | 0.5073 | 0.1928 | 1.748 |
| Technical Support | 2 | 1 | 1 | 50.0% | 0.3055 | 0.0946 | 1.263 |

## Accuracy Per Difficulty

| Difficulty | N | Pass | Fail | Accuracy | Avg WER | Avg CER | Avg Time (s) |
|---|---|---|---|---|---|---|---|
| Easy | 14 | 6 | 8 | 42.9% | 0.3925 | 0.1405 | 1.311 |
| Hard | 14 | 5 | 9 | 35.7% | 0.436 | 0.2009 | 2.562 |
| Medium | 32 | 11 | 21 | 34.4% | 0.5297 | 0.2706 | 1.651 |

---

## Full Results Table

| ID | Language | Category | Difficulty | Expected | Recognized | WER | CER | Sim | Time(s) | Conf | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EN-01 | English | Greetings | Easy | Hello, good morning, is this Maga Maharaja Foods? | hello good morning is dis Maga Maharaja foods | 0.125 | 0.0513 | 0.967 | 3.241 | 0.6943 | PASS |
| EN-02 | English | Business Hours | Easy | What are your business hours on Sunday? | what R business on Sunday | 0.4286 | 0.3438 | 0.7937 | 1.062 | 0.7815 | FAIL |
| EN-03 | English | Pricing | Medium | How much does the five hundred gram mango pickle jar cost? | how Mach Dus D 500 gram mango pickle jar cost | 0.4545 | 0.3404 | 0.7647 | 1.529 | 0.5613 | FAIL |
| EN-04 | English | Product Information | Medium | Do you have any sugar-free pickle options available? | DU u have any sugar free pickle options available | 0.5 | 0.0698 | 0.9495 | 1.639 | 0.7383 | FAIL |
| EN-05 | English | Delivery | Medium | Can you deliver to Gachibowli by tomorrow evening? | can u deliver Tu Gucci boli bye tomorrow evening | 0.625 | 0.1667 | 0.8866 | 1.59 | 0.5545 | FAIL |
| EN-06 | English | Order Status | Medium | I placed an order two days ago, why hasn't it shipped yet? | I placed on order Tu days | 0.6667 | 0.5909 | 0.575 | 2.371 | 0.7144 | FAIL |
| EN-07 | English | Complaint Handling | Hard | I ordered three jars of gongura pickle last Tuesday and only two arrived, and one of them was leaking all over the box, so I want either a refund or a replacement sent immediately. | I order three charge off Ganga pickle last Tuesday and only Tu arrived and Van off Dem vause licking all over D box so I want either an refund or an replacement sent immediately | 0.3824 | 0.1875 | 0.8927 | 4.028 | 0.6408 | FAIL |
| EN-08 | English | Numbers | Medium | I would like to order twelve packets of masala mixture, please. | I Wood like Tu order 12 packets off masala mixture please | 0.3636 | 0.1961 | 0.8814 | 1.298 | 0.6855 | FAIL |
| EN-09 | English | Addresses | Hard | Please deliver it to flat number four B, Sri Sai Residency, near Ayyappa Society, Madhapur, Hyderabad, five hundred zero eight one. | please deliver Tu flat number fore bi Shri Sai Residency near Ayyappa society Madhapur Hyderabad 50081 | 0.4762 | 0.2857 | 0.8194 | 2.582 | 0.5829 | FAIL |
| EN-10 | English | Phone Numbers | Medium | You can reach me at nine eight four zero one two three four five six. | u can reach me at 9840 123456 | 0.7333 | 0.7593 | 0.3918 | 1.761 | 0.5953 | FAIL |
| HI-01 | Hindi | Greetings | Easy | Namaste ji, aap kaise hain? | namaste ji aap kaise hain | 0.0 | 0.0 | 1.0 | 0.754 | 0.931 | PASS |
| HI-02 | Hindi | General Conversation | Easy | Aajkal mausam bahut garam ho raha hai na? | aajkal Mausam bahut garm ho raha hai na | 0.125 | 0.0303 | 0.9873 | 1.03 | 0.7569 | PASS |
| HI-03 | Hindi | Customer Support | Easy | Mujhe customer support se baat karni hai. | Mujhe customer support se baat karni hai | 0.0 | 0.0 | 1.0 | 0.998 | 0.968 | PASS |
| HI-04 | Hindi | Order Status | Medium | Mera order abhi tak deliver kyun nahi hua? | Mera order abhi tak deliver Kyon Nahin hua | 0.25 | 0.0588 | 0.9639 | 0.938 | 0.7311 | PASS |
| HI-05 | Hindi | Business Hours | Easy | Aapka office kitne baje khulta hai? | aapka office kitne baje khulata hai | 0.1667 | 0.0345 | 0.9855 | 0.829 | 0.8185 | PASS |
| HI-06 | Hindi | Pricing | Medium | Ek kilo wale gongura achar ka price kya hai? | 1 kilo wale gunguna Achar ka price kya | 0.3333 | 0.2 | 0.8642 | 1.551 | 0.6953 | PASS |
| HI-07 | Hindi | Product Information | Medium | Kya aapke paas bina lehsun wala achar milta hai? | kya aapke pass Bina lahsun wala Achar Milta Hai | 0.2222 | 0.0513 | 0.9574 | 1.038 | 0.7895 | PASS |
| HI-08 | Hindi | Delivery | Medium | Kya aap Kondapur area mein delivery karte hain? | kya Kondapur area Mein delivery Karte Hain | 0.125 | 0.0769 | 0.9545 | 1.064 | 0.7019 | PASS |
| HI-09 | Hindi | Contact Information | Easy | Aapka customer care number kya hai? | aapka customer care number kya hai | 0.0 | 0.0 | 1.0 | 0.875 | 0.9694 | PASS |
| HI-10 | Hindi | FAQ | Medium | Agar order cancel karna ho to kya karna padega? | hai agar order cancel karna ho to kya karna padega | 0.1111 | 0.0789 | 0.9583 | 1.332 | 0.819 | PASS |
| HI-11 | Hindi | Technical Support | Medium | App mein payment karte waqt error aa raha hai. | app mein payment karte waqt error a raha hai | 0.1111 | 0.027 | 0.9888 | 1.307 | 0.6275 | PASS |
| HI-12 | Hindi | Complaint Handling | Hard | Maine pichle hafte teen jar avakaya achar order kiya tha, lekin ek jar toota hua aaya aur dabbe mein tel bhar gaya tha, isliye mujhe refund chahiye. | men Pichhle Hafte 3000 order kiya tha lekin ek Jad Tuta Hua aaya aur dabbe Mein Tel Bhar gaya tha isliye Mujhe refund chahie | 0.3333 | 0.2269 | 0.8625 | 3.291 | 0.5858 | PASS |
| HI-13 | Hindi | Follow-up Questions | Medium | Aapne bola tha kal tak delivery ho jayegi, ab kya update hai? | aapane bola tha cal Tak delivery ho jayegi ab kya update hai | 0.1667 | 0.0417 | 0.9748 | 2.463 | 0.8004 | PASS |
| HI-14 | Hindi | Numbers | Medium | Mujhe do sau gram wale paanch packet chahiye. | Mujhe to 100 gram wale Panch packet chahie | 0.5 | 0.1622 | 0.8837 | 1.33 | 0.5322 | FAIL |
| HI-15 | Hindi | Dates | Medium | Mera order pandrah tareekh ko diya tha. | Mera order 15 tarikh ko diya tha | 0.2857 | 0.2812 | 0.8286 | 1.091 | 0.8709 | PASS |
| HI-16 | Hindi | Addresses | Hard | Yeh parcel Plot number saat, Road number bees, Banjara Hills, Hyderabad bhej dijiye. | yah parcel plot number 7 Road Number 20 Banjara Hills Hyderabad bhej dijiye | 0.2308 | 0.1324 | 0.9161 | 1.739 | 0.6793 | PASS |
| HI-17 | Hindi | Phone Numbers | Medium | Mera number hai nau, aath, saat, chhe, paanch, chaar, teen, do, ek, zero. | mera number hai 9 8 7 6 5 4 3 2 1 0 | 0.7692 | 0.7451 | 0.5102 | 2.442 | 0.4656 | FAIL |
| HI-18 | Hindi | Mixed English+Hindi | Medium | Mujhe apna order track karna hai, please batao ki current status kya hai. | mujhe apna order track karna hai please batao ki current status kya hai | 0.0 | 0.0 | 1.0 | 1.433 | 0.9608 | PASS |
| HI-19 | Hindi | Order Status | Hard | Maine jo order Monday ko kiya tha wo abhi tak track nahi ho raha, matlab website pe pending dikha raha hai teen din se. | men Jo order Mande ko kiya tha vah abhi tak track Nahin ho raha matlab website pending dikha raha hai teen din se | 0.2083 | 0.1277 | 0.9217 | 2.164 | 0.7798 | PASS |
| HI-20 | Hindi | Pricing | Easy | Sabse sasta pickle kaunsa hai? | sabse Sasta Pipal kaun sa hai | 0.6 | 0.12 | 0.8966 | 0.988 | 0.6211 | FAIL |
| HI-21 | Hindi | Product Information | Hard | Kya aapka Kotha Avakaya achar bilkul traditional Andhra style mein banaya jata hai, ya usme koi preservative bhi dalte hain? | kya aapka Kotha Ka Achar bilkul traditional Andhra style Mein banaya jata hai ya usmein koi preservative bhi dalte hain | 0.1 | 0.068 | 0.971 | 2.085 | 0.7924 | PASS |
| HI-22 | Hindi | Delivery | Hard | Agar main abhi order karu to kya kal shaam tak Gachibowli mein mil jayega, ya phir do din lagenge? | hai agar men abhi order Karun to kya cal Sham Tak gachiboli Mein mil Jaega ya FIR do din lagenge | 0.4211 | 0.1538 | 0.9167 | 2.33 | 0.5669 | FAIL |
| HI-23 | Hindi | Complaint Handling | Medium | Jo product mujhe mila wo expiry date ke bahut kareeb tha. | Jo product Mujhe Mila vah expiry date ke bahut Kareeb tha | 0.0909 | 0.0652 | 0.9558 | 1.271 | 0.8011 | PASS |
| HI-24 | Hindi | General Conversation | Medium | Aapka business kab se shuru hua tha? | aapka business kab se shuru hua tha | 0.0 | 0.0 | 1.0 | 0.988 | 0.9801 | PASS |
| HI-25 | Hindi | Mixed English+Hindi | Hard | Actually mujhe apna previous order modify karna hai, kya customer support team se directly baat ho sakti hai abhi? | actually mujhe apna previous order modify karna hai kya customer support team se directly baat ho sakti hai abhi | 0.0 | 0.0 | 1.0 | 2.074 | 0.9606 | PASS |
| TE-01 | Telugu | Greetings | Easy | Namaskaram andi, meeru bagunnara? | namaskar aunty Meru Bagunnara | 0.75 | 0.1786 | 0.8667 | 1.03 | 0.3822 | FAIL |
| TE-02 | Telugu | General Conversation | Easy | Ivala vaathavaranam chala vedi ga undi kada? | yah wala vatavaran chalave Di gown | 1.0 | 0.4324 | 0.6753 | 1.166 | 0.1209 | FAIL |
| TE-03 | Telugu | Customer Support | Easy | Nenu customer support tho maatladali. | Nano customer support to Matra Dali | 0.8 | 0.1562 | 0.8451 | 0.985 | 0.6056 | FAIL |
| TE-04 | Telugu | Order Status | Medium | Na order eppudu delivery avutundi? | ho na auto epudu delivery Ave tundi | 1.0 | 0.3103 | 0.7941 | 1.887 | 0.2007 | FAIL |
| TE-05 | Telugu | Business Hours | Easy | Mee office timings enti? | MI office timings in | 0.5 | 0.25 | 0.8372 | 2.104 | 0.3815 | FAIL |
| TE-06 | Telugu | Pricing | Medium | Mee pricing details cheppandi, oka kg gongura pachadi ki entha? | MI pricing details check Pandi Okha kilo gram gongura Pachadi ki n tha | 0.8 | 0.2308 | 0.8702 | 1.92 | 0.4414 | FAIL |
| TE-07 | Telugu | Product Information | Medium | Meeku vellulli lekunda pachadi dorukutunda? | main ko value Lekar dhundha | 1.0 | 0.6842 | 0.4348 | 1.655 | 0.1308 | FAIL |
| TE-08 | Telugu | Delivery | Medium | Meeru Kondapur area ki delivery chestara? | Neeru Kondapur area ke delivery Chittara | 0.5 | 0.1143 | 0.9 | 1.221 | 0.4697 | FAIL |
| TE-09 | Telugu | Contact Information | Easy | Mee customer care number enti? | MI customer care number anti | 0.4 | 0.12 | 0.9123 | 1.127 | 0.6406 | FAIL |
| TE-10 | Telugu | FAQ | Medium | Order cancel cheyalante em cheyali? | order cancel challenge | 0.6 | 0.4333 | 0.6786 | 1.127 | 0.7043 | FAIL |
| TE-11 | Telugu | Technical Support | Medium | App lo payment chesetappudu error vastundi. | aap Lo payment Jaise Tappu error vastundi | 0.5 | 0.1622 | 0.8675 | 1.219 | 0.4018 | FAIL |
| TE-12 | Telugu | Complaint Handling | Hard | Nenu last week moodu jar avakaya pachadi order chesanu, kaani oka jar pagilipoyi undi mariyu box antha noone poyindi, kabatti nenu refund kavali. | Nenu last week module jarake aap pachas kahaniyon ka jar bagheli poi Undi Mari box anthon Ne poeam Di Kabaddi Nenu refund kavvali | 0.7391 | 0.325 | 0.7823 | 2.916 | 0.1372 | FAIL |
| TE-13 | Telugu | Follow-up Questions | Medium | Meeru repu ki delivery avutundi ani cheppaaru, ippudu em update undi? | Meru Repo ki delivery Avva Tu yani Chhe Paru eppude update | 0.8182 | 0.2982 | 0.8 | 1.954 | 0.1818 | FAIL |
| TE-14 | Telugu | Numbers | Medium | Naaku rendu vandala gram unna five packets kavali. | na koreno Wonderla gram Unnao five packets kavvali | 0.625 | 0.2381 | 0.8283 | 3.96 | 0.4959 | FAIL |
| TE-15 | Telugu | Dates | Medium | Na order padihenu tarikuna pettanu. | Nau Abadi hai no daddy cool upan | 1.4 | 0.7 | 0.3636 | 1.259 | 0.0418 | FAIL |
| TE-16 | Telugu | Addresses | Hard | Ee parcel ni Plot number edu, Road number iravai, Banjara Hills, Hyderabad ki pampandi. | yah parcel Ne plot number road number iravai Banjara Hills Hyderabad | 0.3571 | 0.2429 | 0.8477 | 2.049 | 0.5935 | FAIL |
| TE-17 | Telugu | Phone Numbers | Medium | Naa number tommidi, enimidi, edu, aaru, aidu, naalugu, moodu, rendu, okati, sunna. | na number to midi animi Di ADO r o i DU na logo Modu Rendu okati sunna | 1.0833 | 0.2131 | 0.831 | 2.613 | 0.1703 | FAIL |
| TE-18 | Telugu | Mixed English+Telugu | Medium | Naaku na order track cheyali, current status enti ani cheppandi. | nakhun aur attract 46 current status India | 0.8 | 0.5094 | 0.5962 | 2.735 | 0.3609 | FAIL |
| TE-19 | Telugu | Order Status | Hard | Nenu Monday roju pettina order ippatiki track avvatledu, website lo three days nunchi pending ani chupistundi. | Nenu Monday Roju peti na order utpati ki track avat ledu website Lo 3 days Munshi pending yani chupi | 0.625 | 0.1935 | 0.875 | 3.145 | 0.2458 | FAIL |
| TE-20 | Telugu | Pricing | Easy | Anni pachadi lo cheapest edi? | Anya Pachadi Lo chipest ad | 0.6 | 0.25 | 0.8148 | 2.164 | 0.2036 | FAIL |
| TE-21 | Telugu | Product Information | Hard | Mee Kotha Avakaya pachadi bilkul traditional Andhra style lo chestara, leda preservatives kuda vestara? | MI ko tha avakaya bachari bilkul traditional Andhra style Lo chest Tara ledha preservative Skoda vistara | 0.7143 | 0.0909 | 0.9171 | 2.325 | 0.2859 | FAIL |
| TE-22 | Telugu | Delivery | Hard | Ippudu order chesthe repu sayantram Gachibowli lo dorukutunda, leda rendu rojulu paduthunda? | ko epudu orders the Repo sayantram Gachibowli Lodo ruko Tunda | 0.9167 | 0.4051 | 0.6887 | 2.329 | 0.0366 | FAIL |
| TE-23 | Telugu | Complaint Handling | Medium | Naaku vachina product expiry date daggaraga undi. | na ko Vachan product expiry date | 0.7143 | 0.4048 | 0.725 | 1.63 | 0.4499 | FAIL |
| TE-24 | Telugu | General Conversation | Medium | Mee business eppudu start ayyindi? | main business app Tu start | 0.8 | 0.4483 | 0.678 | 1.212 | 0.4108 | FAIL |
| TE-25 | Telugu | Mixed English+Telugu | Hard | Actually naaku na previous order modify cheyali, customer support team tho direct ga maatladagalama ippudu? | actually knock on an previous order modified 46 customer support team for direct produ | 0.6 | 0.3736 | 0.7225 | 2.818 | n/a | FAIL |