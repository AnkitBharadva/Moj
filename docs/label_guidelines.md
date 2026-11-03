# Comment Labeling Guidelines (@cockroachjantaparty Dataset)

## Overview
Each comment is labeled on three independent dimensions:
1. **Sentiment**: `positive` / `neutral` / `negative`
2. **Hate**: `hateful` / `offensive but not hateful` / `neither`
3. **Support/Stance**: `supports` / `opposes` / `neither` (target: Cockroach Janta Party / activist campaign)

---

## 1. Sentiment

- **positive**: Expresses approval, praise, happiness, encouragement, appreciation.
  - *"Congratulations all of you 🙌❤️"*
  - *"Full support CJP"*
  - *"salute to CJP and the team..atleast they are trying to cover up and show to the nation"*
- **neutral**: Factual queries, neutral tags, reporting administrative/exam details without overt emotional bias.
  - *"Delhi police driver exam record code: 5555"*
  - *"What time is the live stream starting?"*
- **negative**: Expresses disapproval, anger, cynicism, frustration, sorrow, opposition.
  - *"India now in wrost condition ever historical period"*
  - *"Sab jumla hai.. kuch nhi hoga"*
  - *"Never trust politicians!"*
  - *"Being a student is a Joke in this country 😢"*

---

## 2. Hate Speech & Abuse

- **hateful**: Attacks, dehumanizes, threatens, or incites hatred/violence against an identity group (based on caste, religion, gender, sexual orientation, disability, or ethnic group like Adivasis).
  - Calling for boycotts or violence against religious minorities or marginalized groups.
- **offensive but not hateful**: Vulgarity, profanity, crude insults aimed at specific individuals, parties, or institutions without targeting a protected identity group.
  - *"ye bta bhadwe tmko kaun sa post mili bartan saf krne ka"*
  - *"LAUDA ka cjp 😂"*
  - *"chutiya politicians"*
- **neither**: Negative political criticism, satire, or protest language that is free of vulgar slurs and group discrimination.
  - *"Yogi dar Gaya tha pr 😂"*
  - *"BJP hatao Desh bachao"*
  - *"Modi It's done bro & Take your JHOLA"*

---

## 3. Support / Stance (Target: CJP / Featured Activist Issue)

- **supports**: Explicitly or implicitly agrees with CJP's campaign, praises their team, or echoes their cause (e.g., student justice, tribal school repairs).
  - *"C J P respect button 💖"*
  - *"Adivasi school thik karo"*
  - *"Hum sab milkr thik karwake hi rahenge adivasi school thik."*
  - *"CJP Party' aura Day by day 🔥"*
- **opposes**: Attacks CJP, supports opponent platforms in direct opposition to CJP's narrative, or rejects CJP's motives.
  - *"Only BJP 🚩"*
  - *"LAUDA ka cjp"*
  - *"Guys pls enough with the AI generated posts"*
- **neither**: Off-topic, spam, general commentary that does not express a stance toward CJP or its activist focus.
  - *"Mujhe bhi rakh lo 👏👏👏"*
  - *"What is the syllabus for this exam?"*

---

## 4. Tricky Cases & Nuances

### Sarcasm & Engagement Memes
> *"Narendra Modi resign button 👇"*
- **Sentiment**: `negative` (anti-incumbent)
- **Hate**: `neither` (political protest meme)
- **Support**: `supports` (aligns with CJP's oppositional activist agenda)

### Quoted Hate or Injustice Reporting
> *"A candidate paid 20 lakh bribe and said 'go back' to students"*
- **Sentiment**: `negative`
- **Hate**: `neither` (reporting corruption/abuse, not promoting hate)
- **Support**: `supports` (supporting whistleblower/reform)

### Hinglish Cynicism
> *"Sab jumla hai.. kuch nhi hoga"*
- **Sentiment**: `negative`
- **Hate**: `neither`
- **Support**: `opposes` (dismissive of CJP's activism impact)

### Mixed Sentiment with Supportive Action
> *"This is heartbreaking and disgusting, we stand with the students!"*
- **Sentiment**: `negative` (emotional distress)
- **Hate**: `neither`
- **Support**: `supports`
