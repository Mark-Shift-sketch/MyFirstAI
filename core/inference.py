import re
import html
import datetime
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
import json
import difflib
from pathlib import Path

WEB_SEARCH_ENDPOINT = "https://duckduckgo.com/html/?q={query}"
WEB_TIMEOUT_SECONDS = 6
WEB_MAX_SOURCE_CHARS = 0
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

KNOWLEDGE_PATH = Path("data/offline_knowledge.json")

OFFLINE_CAPITALS = {
    "japan": "Tokyo", "france": "Paris", "germany": "Berlin", "india": "New Delhi",
    "united states": "Washington, D.C.", "usa": "Washington, D.C.", "united kingdom": "London",
    "uk": "London", "canada": "Ottawa", "china": "Beijing", "brazil": "Brasilia",
    "australia": "Canberra", "italy": "Rome", "spain": "Madrid", "russia": "Moscow",
}

OFFLINE_PEOPLE = {
    "alan turing": "Alan Turing was a mathematician and computer science pioneer who laid foundations for modern computing and AI.",
    "albert einstein": "Albert Einstein was a physicist best known for the theory of relativity and contributions to modern physics.",
    "isaac newton": "Isaac Newton was a physicist and mathematician known for laws of motion, gravity, and calculus development.",
    "nikola tesla": "Nikola Tesla was an inventor and engineer known for AC power systems and electrical innovation.",
    "ada lovelace": "Ada Lovelace is often regarded as the first computer programmer for her work on Charles Babbage's Analytical Engine.",
}

OFFLINE_FACTS = {
    "what is ai": "AI stands for artificial intelligence, which is the field of creating systems that perform tasks requiring human-like intelligence.",        
    "what is machine learning": "Machine learning is a branch of AI where models learn patterns from data to make predictions or decisions.",
    "what is deep learning": "Deep learning is a machine learning approach using neural networks with many layers to learn complex patterns.",
    "what is python": "Python is a high-level programming language known for readability and a large ecosystem for web, automation, data science, and AI.",     
    "what is programming": "Programming is the process of writing instructions that tell a computer how to perform tasks.",
    "what is github": "GitHub is a platform for hosting and collaborating on code using Git version control.",
    "what is internet": "The internet is a global network of interconnected computers that share information using standard protocols.",
    "what is cpu": "A CPU is the central processing unit, often called the brain of the computer, responsible for executing instructions.",
    "what is ram": "RAM is temporary memory that stores active data for fast access while your computer is running.",
    "what is an algorithm": "An algorithm is a step-by-step method for solving a problem or performing a computation.",
}

CONVERSATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "the", "to", "we", "what", "when", "where", "which",
    "who", "why", "you", "your",
}

FACT_MATCH_STOPWORDS = {
    "what", "is", "who", "are", "the", "a", "an", "of", "tell", "me", "about", "define", "explain",
}

QA_HIGH_CONFIDENCE = 0.72
QA_MEDIUM_CONFIDENCE = 0.56
QA_LOW_CONFIDENCE = 0.40
RAG_MIN_GROUNDING_SCORE = 0.45


class InferenceManager:
    def __init__(self, jarvis):
        self.jarvis = jarvis
        self.web_cache = {}
        self.offline_knowledge = self._load_offline_knowledge()

    def _dataset_qa_answer(self, text):
        if not text.strip():
            return None

        normalized = self.jarvis._normalize_qa_key(text)
        if not normalized:
            return None

        direct = self.jarvis.qa_lookup.get(normalized)
        if direct:
            return direct

        compact = normalized.replace(" ", "")
        for key, value in self.jarvis.qa_lookup.items():
            if compact == key.replace(" ", ""):
                return value

        if not self.jarvis.qa_questions:
            return None

        close = difflib.get_close_matches(normalized, self.jarvis.qa_questions, n=1, cutoff=0.78)
        if close:
            return self.jarvis.qa_lookup.get(close[0])

        best_key = ""
        best_score = 0.0
        for key in self.jarvis.qa_questions:
            ratio = difflib.SequenceMatcher(None, normalized, key).ratio()      
            overlap = self._token_overlap_score(normalized, key)
            score = (0.6 * ratio) + (0.4 * overlap)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= 0.72:
            return self.jarvis.qa_lookup.get(best_key)

        return None


    def _semantic_qa_response(self, text):
        match = self.brain.get_match(text)
        if not match:
            return None

        answer = str(match.get("answer", "")).strip()
        matched_question = str(match.get("question", "")).strip()
        score = float(match.get("score", 0.0))

        if not answer or not matched_question:
            return None

        if score >= QA_HIGH_CONFIDENCE:
            return answer

        if score >= QA_MEDIUM_CONFIDENCE:
            return f"{answer} Does this match what you meant?"

        if score >= QA_LOW_CONFIDENCE:
            return (
                f"I might be close, but I am not fully sure. "
                f"Are you asking about {matched_question}?"
            )

        return None


    @staticmethod
    def _verify_grounding(draft, facts):
        draft_tokens = set(re.findall(r"[a-z0-9']+", str(draft or "").lower())) 
        if not draft_tokens:
            return False
        fact_tokens = set()
        for fact in facts:
            fact_tokens.update(re.findall(r"[a-z0-9']+", str(fact).lower()))    
        if not fact_tokens:
            return False

        overlap = len(draft_tokens & fact_tokens)
        ratio = overlap / max(len(draft_tokens), 1)
        return ratio >= 0.12


    def _planner_response(self, text):
        intent = self.jarvis._detect_intent(text)
        matches = self.brain.get_top_matches(text, k=3, min_score=RAG_MIN_GROUNDING_SCORE)
        grounded_facts = [item.get("answer", "") for item in matches if item.get("answer")]

        # Use deterministic local reasoners as additional grounded context.     
        factual = self.jarvis._knowledge_answer(text)
        if factual:
            grounded_facts.insert(0, factual)

        if self.jarvis._is_question(text) and not grounded_facts:
            uncertain = (
                "I am not fully certain based on current grounded data. "       
                "Could you clarify the topic or teach me that fact?"
            )
            self._update_session_memory(text, uncertain, intent=intent, facts=[])
            return uncertain

        draft = self.rag_generator.generate(
            question=text,
            facts=grounded_facts,
            intent=intent,
        )
        if not draft:
            if grounded_facts:
                draft = grounded_facts[0]
            else:
                draft = "I am not fully certain based on current grounded data."

        if grounded_facts and not self._verify_grounding(draft, grounded_facts):
            final = (
                "I may be missing reliable grounding for that. "
                "Please confirm your target topic and I will answer precisely." 
            )
            self._update_session_memory(text, final, intent=intent, facts=grounded_facts)
            return final

        final = self.jarvis._compact_answer(draft, max_chars=420)
        self._update_session_memory(text, final, intent=intent, facts=grounded_facts)
        return final


    @staticmethod
    def _is_likely_search_result_url(url):
        if not url or not isinstance(url, str):
            return False
        if not url.startswith(("http://", "https://")):
            return False

        host = urlparse(url).netloc.lower()
        blocked_hosts = (
            "duckduckgo.com",
            "youtube.com",
            "facebook.com",
            "instagram.com",
        )
        return not any(blocked in host for blocked in blocked_hosts)


    def _http_get_text(self, url, timeout=WEB_TIMEOUT_SECONDS, max_bytes=350_000):
        request = Request(url, headers={"User-Agent": WEB_USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes)
            charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="ignore")


    @staticmethod
    def _html_to_text(fragment):
        text = re.sub(r"<script[\s\S]*?</script>", " ", fragment, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    @staticmethod
    def _strip_inline_citations(text):
        cleaned = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", str(text or ""))  
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


    def _extract_search_links(self, search_html, limit=5):
        links = []
        for href in re.findall(r'href="([^"]+)"', search_html, flags=re.IGNORECASE):
            candidate = html.unescape(href)
            if "uddg=" in candidate:
                encoded = candidate.split("uddg=", 1)[1].split("&", 1)[0]       
                candidate = unquote(encoded)

            if candidate.startswith("//"):
                candidate = "https:" + candidate

            if not self._is_likely_search_result_url(candidate):
                continue

            if candidate not in links:
                links.append(candidate)

            if len(links) >= limit:
                break

        # Prefer encyclopedic sources when available.
        links.sort(key=lambda link: ("wikipedia.org" not in link, "britannica.com" not in link))
        return links


    def _extract_page_summary(self, page_html):
        title_match = re.search(r"<title[^>]*>(.*?)</title>", page_html, flags=re.IGNORECASE | re.DOTALL)
        page_title = self._html_to_text(title_match.group(1)) if title_match else ""

        page_html = re.sub(r"<script[\s\S]*?</script>", " ", page_html, flags=re.IGNORECASE)
        page_html = re.sub(r"<style[\s\S]*?</style>", " ", page_html, flags=re.IGNORECASE)
        paragraphs = re.findall(r"<p[^>]*>([\s\S]*?)</p>", page_html, flags=re.IGNORECASE)

        best = ""
        for paragraph in paragraphs:
            cleaned = self._html_to_text(paragraph)
            if len(cleaned) < 90:
                continue
            if "cookie" in cleaned.lower() and len(cleaned) < 200:
                continue
            best = cleaned
            break

        if not best:
            text_only = self._html_to_text(page_html)
            if len(text_only) >= 140:
                best = text_only

        return page_title, best


    @staticmethod
    def _normalize_web_query(raw_question):
        tokens = re.findall(r"[a-zA-Z0-9']+", str(raw_question or "").strip())  
        if not tokens:
            return ""

        lowered = [token.lower() for token in tokens]
        if len(lowered) >= 2 and lowered[0] == "what" and lowered[1] == "s":    
            lowered[1] = "is"

        rebuilt = []
        index = 0
        while index < len(lowered):
            token = lowered[index]
            if len(token) == 1 and token.isalpha():
                probe = index
                while probe < len(lowered) and len(lowered[probe]) == 1 and lowered[probe].isalpha():
                    probe += 1

                if probe - index >= 2:
                    rebuilt.append("".join(lowered[index:probe]).upper())
                else:
                    rebuilt.extend(lowered[index:probe])
                index = probe
                continue

            rebuilt.append(token)
            index += 1

        return " ".join(rebuilt).strip()


    def _web_answer(self, user_question):
        normalized_key = self.jarvis._normalize_qa_key(user_question)
        if normalized_key in self.web_cache:
            return self.web_cache[normalized_key]

        query = self._normalize_web_query(user_question).strip(" ?")
        if not query:
            return None

        search_url = WEB_SEARCH_ENDPOINT.format(query=quote_plus(query))
        try:
            from urllib.request import Request, urlopen
            import html, re
            request = Request(search_url, headers={"User-Agent": WEB_USER_AGENT})
            with urlopen(request, timeout=WEB_TIMEOUT_SECONDS) as response:
                search_html = response.read(350000).decode("utf-8", errors="ignore")

            snippets = re.findall(r'class="result__snippet[^>]*>(.*?)</a>', search_html, flags=re.IGNORECASE | re.DOTALL)
            if not snippets:
                return None

            clean_snippets = []
            for s in snippets[:3]:
                txt = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
                if txt and len(txt) > 30 and 'cookie' not in txt.lower():
                    clean_snippets.append(txt)

            if not clean_snippets:
                return None

            draft = self.jarvis.rag_generator.generate(
                question=user_question,
                facts=clean_snippets,
                intent="informational"
            )

            if not draft or "not fully certain" in draft.lower() or len(draft) < 10:
                draft = clean_snippets[0]

            final = self.jarvis._compact_answer(draft, max_chars=400)
            self.web_cache[normalized_key] = final
            return final

        except Exception as e:
            return None

    def _load_offline_knowledge(self):
        default_data = {"facts": {}, "people": {}, "capitals": {}}
        if not KNOWLEDGE_PATH.exists():
            return default_data

        try:
            with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file_handle:     
                loaded = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_data

        if not isinstance(loaded, dict):
            return default_data

        cleaned = {}
        for key in ("facts", "people", "capitals"):
            bucket = loaded.get(key, {})
            if not isinstance(bucket, dict):
                bucket = {}
            cleaned[key] = {
                self.jarvis._normalize_key(str(k)): str(v).strip()
                for k, v in bucket.items()
                if str(k).strip() and str(v).strip()
            }

        return cleaned


    def _save_offline_knowledge(self):
        KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with KNOWLEDGE_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.offline_knowledge, file_handle, indent=2)


    @staticmethod
    def _extract_followup_subject(text):
        normalized = text.strip().lower().strip(" ?!.")
        normalized = re.sub(r"^(?:and|about|what about|how about)\s+", "", normalized)
        if not normalized:
            return ""
        if re.fullmatch(r"[a-z\s.]{2,60}", normalized):
            return normalized.replace(".", "").strip()
        return ""


    @staticmethod
    def _token_overlap_score(left, right):
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / max(union, 1)


    @staticmethod
    def _detect_mood(text):
        lowered = text.strip().lower()
        if re.search(r"\b(sad|stressed|tired|upset|anxious|overwhelmed|frustrated|angry|burned out)\b", lowered):
            return "negative"
        if re.search(r"\b(happy|great|good|excited|motivated|confident|calm|relaxed)\b", lowered):
            return "positive"
        if re.search(r"\b(confused|unsure|not sure|don't know|do not know|stuck)\b", lowered):
            return "uncertain"
        return "neutral"


    @staticmethod
    def _extract_goal_phrase(text):
        lowered = text.strip().lower()
        patterns = [
            r"\bi want to\s+(.+)$",
            r"\bi need to\s+(.+)$",
            r"\bmy goal is to\s+(.+)$",
            r"\bhelp me\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return match.group(1).strip(" .")
        return ""


    @staticmethod
    def _extract_focus_phrase(text):
        lowered = text.strip().lower()
        explicit_match = re.search(
            r"(?:about|on|for|with)\s+([a-z0-9'\-\s]{3,70})$",
            lowered,
        )
        if explicit_match:
            phrase = explicit_match.group(1).strip(" .")
            if phrase:
                return phrase

        tokens = re.findall(r"[a-z0-9']+", lowered)
        key_tokens = [
            token
            for token in tokens
            if len(token) > 1 and token not in CONVERSATION_STOPWORDS
        ]
        if not key_tokens:
            return ""
        if len(key_tokens) <= 3:
            return " ".join(key_tokens)
        return " ".join(key_tokens[-3:])


    @staticmethod
    def _clean_generated_reply(text):
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return ""

        cleaned = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^a-zA-Z0-9 ,.?!'\-]", "", cleaned)
        cleaned = re.sub(r"^(?:jarvis|assistant)\b[:,\s-]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" .")

        if len(cleaned) < 12:
            return ""

        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned = cleaned + "."
        return cleaned


    def _humanized_fallback(self, text, generated_text=""):
        mood = self._detect_mood(text)
        focus = self._extract_focus_phrase(text)
        generated = self._clean_generated_reply(generated_text)

        if mood == "negative":
            opener = "That sounds like a lot"
            guide = "We can make it easier by picking one small next step"      
        elif mood == "positive":
            opener = "I like that energy"
            guide = "Let's channel it into a clear action"
        elif mood == "uncertain":
            opener = "That uncertainty is normal"
            guide = "We can reduce it by structuring the options"
        else:
            opener = self.jarvis._pick_variant(
                [
                    "I hear you",
                    "I am with you",
                    "Thanks for sharing that",
                    "That makes sense",
                ],
                text,
            )
            guide = "Let's turn this into something practical"

        focus_line = f" We can focus on {focus}." if focus else ""
        if generated:
            idea_line = f" {generated}"
        elif self.jarvis._is_question(text):
            idea_line = " Do you want a quick answer first, or a step-by-step walkthrough?"
        else:
            idea_line = " Tell me the exact outcome you want, and I will help you get there."

        base = f"{opener}.{focus_line} {guide}.{idea_line}".strip()
        return self.jarvis._compact_answer(base, max_chars=360)


    def _followup_knowledge_answer(self, text):
        subject = self._extract_followup_subject(text)
        if not subject:
            return None

        last_intent = self.jarvis.conversation_state.get("last_knowledge_intent", "")
        if not last_intent:
            return None

        if last_intent == "capital":
            capitals = {**OFFLINE_CAPITALS, **self.offline_knowledge.get("capitals", {})}
            value = capitals.get(subject)
            if value:
                self.jarvis._set_knowledge_context("capital", subject)
                return f"The capital of {subject.title()} is {value}."

        if last_intent == "person":
            people = {**OFFLINE_PEOPLE, **self.offline_knowledge.get("people", {})}
            value = people.get(subject)
            if value:
                self.jarvis._set_knowledge_context("person", subject)
                return value

        if last_intent == "definition":
            facts = {**OFFLINE_FACTS, **self.offline_knowledge.get("facts", {})}
            probe_keys = [subject, f"what is {subject}"]
            for key in probe_keys:
                value = facts.get(key)
                if value:
                    self.jarvis._set_knowledge_context("definition", subject)   
                    return value

        return None


    def _fuzzy_fact_answer(self, query):
        facts = {**OFFLINE_FACTS, **self.offline_knowledge.get("facts", {})}    
        normalized_query = self.jarvis._normalize_key(query)

        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", normalized_query)
            if token not in FACT_MATCH_STOPWORDS
        }
        if not query_tokens:
            return None

        best_key = ""
        best_score = 0.0
        for key in facts:
            key_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", key)
                if token not in FACT_MATCH_STOPWORDS
            }
            if not key_tokens:
                continue

            exact_overlap = len(query_tokens & key_tokens)
            if exact_overlap == 0:
                # Allow minor spelling mistakes only when topic words are close.
                near_match = any(
                    difflib.SequenceMatcher(None, left, right).ratio() >= 0.86  
                    for left in query_tokens
                    for right in key_tokens
                )
                if not near_match:
                    continue

            score = self._token_overlap_score(" ".join(sorted(query_tokens)), " ".join(sorted(key_tokens)))
            if score > best_score:
                best_key = key
                best_score = score

        if best_key and best_score >= 0.35:
            return facts[best_key]
        return None


    def _teach_local_knowledge(self, text):
        capital_match = re.search(
            r"\b(?:learn|teach)\s+that\s+the\s+capital\s+of\s+([a-z\s]+)\s+is\s+(.+)$",
            text,
        )
        if capital_match:
            country = self.jarvis._normalize_key(capital_match.group(1))        
            capital = capital_match.group(2).strip(" .").title()
            if not country or not capital:
                return None
            self.offline_knowledge.setdefault("capitals", {})[country] = capital
            self._save_offline_knowledge()
            return f"Learned. The capital of {country.title()} is {capital}."   

        fact_match = re.search(r"\b(?:learn|teach)\s+that\s+(.+?)\s+is\s+(.+)$", text)
        if fact_match:
            subject = self.jarvis._normalize_key(fact_match.group(1))
            definition = fact_match.group(2).strip(" .")
            if not subject or not definition:
                return None

            if len(subject.split()) <= 6:
                key = f"what is {subject}"
            else:
                key = subject

            self.offline_knowledge.setdefault("facts", {})[key] = definition    
            self._save_offline_knowledge()
            return f"Knowledge stored for {subject}."

        return None
