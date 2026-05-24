import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.saving import register_keras_serializable

from app.extractor import clean_text


MAX_TOKENS = 256


def strip_surrogates(text: str) -> str:
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def clean_profile_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = strip_surrogates(text)
    return text.strip()


_TOKEN_PATTERN = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    r"|[a-zA-Z0-9][a-zA-Z0-9._+#/\\-]*"
    r"|[^\s\w]"
)


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def normalize_token_for_vocab(token: str) -> str:
    return re.sub(r"\d", "0", token.lower())


def chunk_tokens(text: str, tokens_info: List[Tuple[str, int, int]], max_tokens: int = MAX_TOKENS):
    if len(tokens_info) <= max_tokens:
        return [tokens_info]

    newline_idx = set()
    for i in range(1, len(tokens_info)):
        if "\n" in text[tokens_info[i - 1][2]: tokens_info[i][1]]:
            newline_idx.add(i)

    chunks = []
    start = 0
    n = len(tokens_info)

    while start < n:
        end = min(start + max_tokens, n)

        if end < n:
            best = end
            for j in range(end, max(start + 1, end - 20) - 1, -1):
                if j in newline_idx:
                    best = j
                    break
            end = best

        chunks.append(tokens_info[start:end])
        start = end

    return chunks


@register_keras_serializable(package="MyModels", name="BiLSTMAttentionModel")
class BiLSTMAttentionModel(Model):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        rnn_units: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="word_embedding",
        )

        self.bilstm = layers.Bidirectional(
            layers.LSTM(
                rnn_units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=0.0,
            ),
            name="bi_lstm",
        )

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=rnn_units,
            dropout=dropout,
            name="multihead_attention",
        )

        self.dense1 = layers.Dense(rnn_units, activation="relu", name="dense_hidden")
        self.dropout_layer = layers.Dropout(dropout, name="dropout")
        self.output_layer = layers.Dense(num_tags, name="tag_logits")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        lstm_out = self.bilstm(x, training=training)

        attn_out = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            training=training,
        )

        x = layers.add([lstm_out, attn_out])
        x = self.dense1(x)
        x = self.dropout_layer(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "num_tags": self.num_tags,
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CandidateProfileEnricher:
    """
    Internal CV profile enrichment layer.

    This layer extracts profile entities from resume text and merges them into
    the existing structured CV representation. It is not exposed as a separate
    scoring output.
    """

    def __init__(self, model_path: Path, vocab_path: Path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            self.vocab = json.load(file)

        self.vocab["idx2tag"] = {
            int(k): v for k, v in self.vocab["idx2tag"].items()
        }

        self.model = keras.models.load_model(
            model_path,
            custom_objects={"BiLSTMAttentionModel": BiLSTMAttentionModel},
            compile=False,
            safe_mode=False,
        )

    def predict_entities(self, text: str) -> List[dict]:
        text = clean_profile_text(text)
        tokens_info = tokenize(text)

        if not tokens_info:
            return []

        chunks = chunk_tokens(text, tokens_info, max_tokens=MAX_TOKENS)
        entities = []

        token_offset = 0
        pad_id = self.vocab["pad_id"]

        for chunk in chunks:
            tokens = [x[0] for x in chunk]
            token_ids = [
                self.vocab["word2idx"].get(
                    normalize_token_for_vocab(token),
                    self.vocab["unk_id"],
                )
                for token in tokens
            ]

            seq_len = min(len(token_ids), MAX_TOKENS)
            token_ids_padded = token_ids[:seq_len] + [pad_id] * (MAX_TOKENS - seq_len)

            input_tensor = tf.constant([token_ids_padded], dtype=tf.int32)

            predictions = self.model(input_tensor, training=False)
            pred_ids = tf.argmax(predictions[0][:seq_len], axis=-1).numpy()

            labels = [
                self.vocab["idx2tag"].get(int(pred_id), "O")
                for pred_id in pred_ids
            ]

            current = None

            for i, (token, label) in enumerate(zip(tokens[:seq_len], labels)):
                global_index = token_offset + i

                if label.startswith("B-"):
                    if current:
                        entities.append(current)

                    current = {
                        "entity": label[2:],
                        "text": token,
                        "start": global_index,
                        "end": global_index,
                    }

                elif label.startswith("I-") and current and current["entity"] == label[2:]:
                    current["text"] += " " + token
                    current["end"] = global_index

                else:
                    if current:
                        entities.append(current)
                        current = None

            if current:
                entities.append(current)

            token_offset += seq_len

        return entities

    def extract(self, resume_text: str) -> dict:
        entities = self.predict_entities(resume_text)
        grouped = self._group_entities(entities)

        return {
            "entities_grouped": grouped,
            "profile_quality_signal": self._calculate_profile_quality_signal(grouped),
            "profile_features": self._calculate_features(grouped),
        }

    def enrich_cv_profile_for_scoring(self, cv_profile: dict, resume_text: str, taxonomy_extractor) -> dict:
        """
        Conservative enrichment for scoring.

        NER output is not used directly.
        Every NER term must be validated again by the existing taxonomy extractor.
        This keeps scoring stable and avoids noisy NER terms becoming skills.
        """

        profile = self.extract(resume_text)
        grouped = profile.get("entities_grouped", {})

        enriched = dict(cv_profile)

        enriched["cv_skills"] = set(enriched.get("cv_skills", set()))
        enriched["cv_tools_technologies"] = set(enriched.get("cv_tools_technologies", set()))
        enriched["cv_domain_keywords"] = set(enriched.get("cv_domain_keywords", set()))
        enriched["cv_project_experience"] = set(enriched.get("cv_project_experience", set()))
        enriched["cv_certifications"] = set(enriched.get("cv_certifications", set()))

        accepted = {
            "skills": set(),
            "tools": set(),
            "domains": set(),
            "projects": set(),
            "certifications": set(),
            "education": "",
            "experience_years": None,
            "seniority": "",
        }

        rejected_terms = []

        candidate_skill_text = " ".join(
            grouped.get("SKILL", [])
            + grouped.get("EXPERTISE", [])
        )

        # Validate NER skill/expertise terms using existing taxonomy.
        ner_skills = taxonomy_extractor._extract(
            candidate_skill_text,
            taxonomy_extractor.skill_re,
            taxonomy_extractor.skill_alias,
        )
        ner_tools = taxonomy_extractor._extract(
            candidate_skill_text,
            taxonomy_extractor.tool_re,
            taxonomy_extractor.tool_alias,
        )
        ner_domains = taxonomy_extractor._extract(
            candidate_skill_text,
            taxonomy_extractor.domain_re,
            taxonomy_extractor.domain_alias,
        )
        ner_projects = taxonomy_extractor._extract(
            candidate_skill_text,
            taxonomy_extractor.project_re,
            taxonomy_extractor.project_alias,
        )
        ner_certs = taxonomy_extractor._extract(
            candidate_skill_text,
            taxonomy_extractor.cert_re,
            taxonomy_extractor.cert_alias,
        )

        accepted["skills"] = ner_skills
        accepted["tools"] = ner_tools | (ner_skills & taxonomy_extractor.tools_vocab)
        accepted["domains"] = ner_domains
        accepted["projects"] = ner_projects
        accepted["certifications"] = ner_certs

        enriched["cv_skills"] |= accepted["skills"]
        enriched["cv_tools_technologies"] |= accepted["tools"]
        enriched["cv_domain_keywords"] |= accepted["domains"]
        enriched["cv_project_experience"] |= accepted["projects"]
        enriched["cv_certifications"] |= accepted["certifications"]

        # Education: only accept if it maps to known education taxonomy.
        education_text = " ".join(grouped.get("EDUCATION", []))
        if education_text:
            education_terms = taxonomy_extractor._extract(
                education_text,
                taxonomy_extractor.education_re,
                taxonomy_extractor.education_alias,
            )
            detected_education = taxonomy_extractor._choose_ranked(
                education_terms,
                taxonomy_extractor.education_rank,
            )

            if detected_education:
                current_rank = taxonomy_extractor.education_rank.get(
                    str(enriched.get("cv_education_level", "")).lower(),
                    -1,
                )
                detected_rank = taxonomy_extractor.education_rank.get(detected_education, -1)

                if detected_rank > current_rank:
                    enriched["cv_education_level"] = detected_education
                    accepted["education"] = detected_education

        # Experience: use only if years can be parsed.
        experience_text = " ".join(grouped.get("EXPERIENCE", []))
        if experience_text:
            ner_years = taxonomy_extractor.extract_years(experience_text)
            current_years = enriched.get("cv_years_experience")

            if self._is_missing_number(current_years) and not self._is_missing_number(ner_years):
                enriched["cv_years_experience"] = ner_years
                accepted["experience_years"] = float(ner_years)
            elif not self._is_missing_number(current_years) and not self._is_missing_number(ner_years):
                enriched["cv_years_experience"] = max(float(current_years), float(ner_years))
                accepted["experience_years"] = float(enriched["cv_years_experience"])

        # Seniority: infer from designation if valid.
        designation_text = " ".join(grouped.get("DESIGNATION", []))
        if designation_text:
            seniority_terms = taxonomy_extractor._extract(
                designation_text,
                taxonomy_extractor.seniority_re,
                taxonomy_extractor.seniority_alias,
            )
            detected_seniority = taxonomy_extractor._choose_ranked(
                seniority_terms,
                taxonomy_extractor.seniority_rank,
            )

            if detected_seniority:
                current_rank = taxonomy_extractor.seniority_rank.get(
                    str(enriched.get("cv_seniority_level", "")).lower(),
                    -1,
                )
                detected_rank = taxonomy_extractor.seniority_rank.get(detected_seniority, -1)

                if detected_rank > current_rank:
                    enriched["cv_seniority_level"] = detected_seniority
                    accepted["seniority"] = detected_seniority

        raw_ner_terms = set()
        for values in grouped.values():
            raw_ner_terms |= {clean_text(value) for value in values if clean_text(value)}

        accepted_all_terms = (
            accepted["skills"]
            | accepted["tools"]
            | accepted["domains"]
            | accepted["projects"]
            | accepted["certifications"]
        )

        rejected_terms = sorted(raw_ner_terms - accepted_all_terms)

        # Conservative profile quality signal.
        # This should not dominate ranking.
        skill_signal = min(len(enriched["cv_skills"]) / 12, 1.0)
        tool_signal = min(len(enriched["cv_tools_technologies"]) / 8, 1.0)
        project_signal = min(len(enriched["cv_project_experience"]) / 3, 1.0)
        education_signal = 1.0 if enriched.get("cv_education_level") else 0.0
        experience_signal = 0.0 if self._is_missing_number(enriched.get("cv_years_experience")) else 1.0
        certification_signal = min(len(enriched["cv_certifications"]) / 3, 1.0)

        profile_quality_signal = round(
            100
            * (
                0.35 * skill_signal
                + 0.20 * tool_signal
                + 0.15 * project_signal
                + 0.10 * education_signal
                + 0.15 * experience_signal
                + 0.05 * certification_signal
            ),
            2,
        )

        enriched["_profile_enrichment"] = {
            "raw_entities": grouped,
            "accepted_for_scoring": {
                "skills": sorted(accepted["skills"]),
                "tools": sorted(accepted["tools"]),
                "domains": sorted(accepted["domains"]),
                "projects": sorted(accepted["projects"]),
                "certifications": sorted(accepted["certifications"]),
                "education": accepted["education"],
                "experience_years": accepted["experience_years"],
                "seniority": accepted["seniority"],
            },
            "rejected_terms": rejected_terms[:100],
            "profile_quality_signal": profile_quality_signal,
            "raw_profile_quality_signal": profile.get("profile_quality_signal"),
            "raw_features": profile.get("profile_features", {}),
        }

        return enriched

    def enrich_cv_profile(self, cv: dict, resume_text: str, taxonomy_extractor) -> dict:
        """
        Merge profile entities into existing cv structure.

        This affects:
        - cv["skills"]
        - cv["tools"]
        - cv["education"]
        - cv["years_experience"]
        - cv["seniority"]
        """

        profile = self.extract(resume_text)
        grouped = profile.get("entities_grouped", {})

        cv = dict(cv)
        cv["skills"] = set(cv.get("skills", set()))
        cv["tools"] = set(cv.get("tools", set()))
        cv["domains"] = set(cv.get("domains", set()))

        skill_terms = self._clean_entity_set(grouped.get("SKILL", []))
        expertise_terms = self._clean_entity_set(grouped.get("EXPERTISE", []))

        # SKILL and EXPERTISE strengthen skill matching.
        cv["skills"] |= skill_terms
        cv["skills"] |= expertise_terms

        # Some extracted skills may also exist as known tools.
        known_tools = {clean_text(x) for x in taxonomy_extractor.tools_vocab}
        cv["tools"] |= {term for term in skill_terms | expertise_terms if term in known_tools}

        # EDUCATION strengthens education matching if taxonomy extractor missed it.
        education_text = " ".join(grouped.get("EDUCATION", []))
        if education_text:
            education_terms = taxonomy_extractor._extract(
                education_text,
                taxonomy_extractor.education_re,
                taxonomy_extractor.education_alias,
            )
            detected_education = taxonomy_extractor._choose_ranked(
                education_terms,
                taxonomy_extractor.education_rank,
            )

            if detected_education:
                current_rank = taxonomy_extractor.education_rank.get(cv.get("education", ""), -1)
                detected_rank = taxonomy_extractor.education_rank.get(detected_education, -1)

                if detected_rank > current_rank:
                    cv["education"] = detected_education

        # EXPERIENCE strengthens years_experience if taxonomy extractor missed it.
        experience_text = " ".join(grouped.get("EXPERIENCE", []))
        if experience_text:
            ner_years = taxonomy_extractor.extract_years(experience_text)
            current_years = cv.get("years_experience")

            if self._is_missing_number(current_years) and not self._is_missing_number(ner_years):
                cv["years_experience"] = ner_years
            elif not self._is_missing_number(current_years) and not self._is_missing_number(ner_years):
                cv["years_experience"] = max(float(current_years), float(ner_years))

        # DESIGNATION may help seniority detection.
        designation_text = " ".join(grouped.get("DESIGNATION", []))
        if designation_text:
            seniority_terms = taxonomy_extractor._extract(
                designation_text,
                taxonomy_extractor.seniority_re,
                taxonomy_extractor.seniority_alias,
            )
            detected_seniority = taxonomy_extractor._choose_ranked(
                seniority_terms,
                taxonomy_extractor.seniority_rank,
            )

            if detected_seniority:
                current_rank = taxonomy_extractor.seniority_rank.get(cv.get("seniority", ""), -1)
                detected_rank = taxonomy_extractor.seniority_rank.get(detected_seniority, -1)

                if detected_rank > current_rank:
                    cv["seniority"] = detected_seniority

        cv["_profile_quality_signal"] = profile["profile_quality_signal"]
        cv["_profile_features"] = profile["profile_features"]

        return cv

    def _group_entities(self, entities: List[dict]) -> Dict[str, List[str]]:
        grouped = {}

        for entity in entities:
            label = str(entity.get("entity", "")).strip().upper()
            value = str(entity.get("text", "")).strip()

            if not label or not value:
                continue

            grouped.setdefault(label, [])

            if value not in grouped[label]:
                grouped[label].append(value)

        return grouped

    def _clean_entity_set(self, values: List[str]) -> set[str]:
        cleaned = set()

        for value in values:
            term = clean_text(value)

            if not term:
                continue

            if len(term) <= 1:
                continue

            # Avoid extremely long phrases becoming noisy skills.
            if len(term.split()) > 6:
                continue

            cleaned.add(term)

        return cleaned

    def _calculate_features(self, grouped: Dict[str, List[str]]) -> dict:
        return {
            "detected_skill_count": len(grouped.get("SKILL", [])),
            "detected_expertise_count": len(grouped.get("EXPERTISE", [])),
            "detected_experience_count": len(grouped.get("EXPERIENCE", [])),
            "detected_education_count": len(grouped.get("EDUCATION", [])),
            "detected_company_count": len(grouped.get("COMPANY", [])),
            "detected_designation_count": len(grouped.get("DESIGNATION", [])),
            "detected_certification_count": len(grouped.get("CERTIFICATION", [])),
            "detected_language_count": len(grouped.get("LANGUAGE", [])),
        }

    def _calculate_profile_quality_signal(self, grouped: Dict[str, List[str]]) -> float:
        features = self._calculate_features(grouped)

        skill_signal = min(features["detected_skill_count"] / 10, 1.0)
        expertise_signal = min(features["detected_expertise_count"] / 6, 1.0)
        experience_signal = min(features["detected_experience_count"] / 4, 1.0)
        education_signal = min(features["detected_education_count"] / 2, 1.0)
        company_signal = min(features["detected_company_count"] / 4, 1.0)
        designation_signal = min(features["detected_designation_count"] / 3, 1.0)
        certification_signal = min(features["detected_certification_count"] / 3, 1.0)
        language_signal = min(features["detected_language_count"] / 3, 1.0)

        score = (
            0.25 * skill_signal
            + 0.15 * expertise_signal
            + 0.20 * experience_signal
            + 0.10 * education_signal
            + 0.10 * company_signal
            + 0.10 * designation_signal
            + 0.06 * certification_signal
            + 0.04 * language_signal
        )

        return round(score * 100, 2)

    def _is_missing_number(self, value) -> bool:
        if value is None:
            return True

        try:
            return bool(np.isnan(value))
        except Exception:
            return False