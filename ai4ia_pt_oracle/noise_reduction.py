"""
NOISE REDUCTION

Feature engineering script to reduce the number of passages that are
guaranteed to NOT have Information Obligations (IOs) — e.g. titles,
subtitles, irrelevant articles, etc.
"""

from typing import Callable, Literal
from dataclasses import dataclass, asdict, field
from enum import auto, StrEnum, Enum
from pydantic import validate_call
import pandera as pa
import pandas as pd
import re
import spacy
from spacy.tokens import Doc as SpacyDoc
from tqdm import tqdm


def normalize_text(df: pd.DataFrame, inputs_key) -> pd.DataFrame:
    """
    - Remove unnecessary leading numbers and letters.
    - Strip leading and trailing white spaces.
    """
    temp_df = df.copy()
    marker_pattern = r"^[0-9]+\s\-\s?|^\d+(?:\.\d+)+\s\-\s?|^[a-z]+\)\s?"
    temp_df[inputs_key] = temp_df[inputs_key].str.replace(
        pat=marker_pattern, repl="", regex=True
    )
    # Apply `strip white spaces`
    temp_df[inputs_key] = temp_df[inputs_key].str.strip()
    return temp_df


@dataclass
class Corpora:
    document: list[str]
    text: list[str]


class CorporaDataFrame(pa.DataFrameModel):
    document: str
    text: str


@dataclass
class CorporaMetadata:
    section_to_keep: Literal["initial", "republished"] | None = None


@dataclass
class Section:
    """Data structure to identify a specific section within a doc.

    Args:
        doc (str): Document's name/code.
        pat (str): Pattern that identifies the section's heading – either regex or literal.
        idx (int, optional): When more than one match returns,
            defines which idx of the matches should be selected. Defaults to 0.
        delimiter (bool, optional): If `True`, all passages following the
            identified heading will be returned. If `False`, only the identified
            section's indices are returned. Defaults to False.
    """

    doc: str
    pat: str
    idx: int = 0
    delimiter: bool = False


class Rules(StrEnum):
    """
    Enumeration class representing different types of rules for noise reduction.
    """

    TITLES = auto()
    SUBTITLES = auto()
    RANDOM_SUBTITLES = auto()
    INTRO_PASSAGES = auto()
    REPUBLISHED_PASSAGES = auto()
    IRRELEVANT_ANNEXES_PASSAGES = auto()
    NO_IO_ARTICLES = auto()
    # IRRELEVANT_MISC_PASSAGES = auto()
    NO_TEXT_PASSAGE = auto()
    MISC_IRELEVANT_PASSAGE = auto()
    ALTERED_PASSAGE = auto()
    ONE_WORD_PASSAGE = auto()
    NOTES_PASSAGE = auto()


@dataclass
class _RuleConfig:
    active: bool
    fn: Callable | None = field(default=None)


@dataclass(frozen=True)
class PatternConfig:
    pat: str
    case: bool  # If `True`, pattern is case sensitive.


class TitlesPatterns(PatternConfig, Enum):
    ARTICLES = r"^«?(ARTIGO\s)?\d+.º(\-[a-z]+)?»?$", False
    ANNEXES = r"^«?(ANEXO|Anexo)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    APPENDICES = r"^«?(APÊNDICE|Apêndice)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    CHAPTERS = (
        r"^«?(CAPÍTULO|Capítulo)(\s+[A-Z0-9]+(.[A-Z0-9]+)?|\s+(ÚNICO|Único))?»?$",
        True,
    )
    BOOKS = r"^«?LIVRO(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    TITLES = r"^«?(TÍTULO|Título)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    DIVISIONS = r"^«?DIVISÃO(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", False
    SUBDIVISIONS = r"^«?SUBDIVISÃO(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", False
    PARTS = r"^«?(PARTE|Parte)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    SECTIONS = r"^«?(SECÇÃO|Secção|SEÇÃO|Seção)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    SUBSECTIONS = r"^«?(SUBSECÇÃO|SUBSEÇÃO)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", False


class CasedTitlesPatterns(PatternConfig, Enum):
    ARTICLES = r"^«?((ARTIGO|Artigo)\s)?\d+.º(\-[A-Za-z]+)?»?$", True
    ANNEXES = r"^«?(ANEXO|Anexo)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    APPENDICES = r"^«?(APÊNDICE|Apêndice)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    BASES = r"^«?(BASE|Base)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    CHAPTERS = (
        r"^«?(CAPÍTULO|Capítulo)(\s+[A-Z0-9]+(.[A-Z0-9]+)?|\s+(ÚNICO|Único))?»?$",
        True,
    )
    BOOKS = r"^«?LIVRO(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    TITLES = r"^«?(TÍTULO|Título)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    DIVISIONS = r"^«?(DIVISÃO|Divisão)(\s[A-Za-z0-9]+(.[A-Za-z0-9]+)?)?»?$", True
    SUBDIVISIONS = r"^«?(SUBDIVISÃO|Subdivisão)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    PARTS = r"^«?(PARTE|Parte)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True
    SECTIONS = r"^«?(SECÇÃO|Secção|SEÇÃO|Seção)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$", True

    # The change to case sensitive for the `SUBSECTIONS` rule actually improved compared to
    # case insensitive. Now it ignores the 'SUBSECÇãO I' wrong instance (it is a subtitle).
    SUBSECTIONS = (
        r"^«?(SUBSECÇÃO|SUBSEÇÃO|Subsecção|Subseção)(\s[A-Z0-9]+(.[A-Z0-9]+)?)?»?$",
        True,
    )


class NoiseReducer:
    _STRUCTURAL_RULES = [
        Rules.TITLES.value,
        Rules.SUBTITLES.value,
        Rules.RANDOM_SUBTITLES.value,
    ]
    _MARKERS_PATTERNS = {"number"}
    _NO_IO_ARTICLES = [
        "Objeto",
        "Objeto e âmbito",
        "Objeto e âmbito de aplicação",
        "Âmbito",
        "Definições",
        "Disposição transitória",
        "Norma transitória",
        "Norma revogatória",
        "Norma final",
        "Produção de efeitos",
        "Entrada em vigor",
        "Entrada em vigor e vigência",
        "Entrada em vigor e produção de efeitos",
        "Informação estatística",
        "Alteração sistemática",
        "Republicação",
        "Legislação subsidiária",
        # "Reavaliação",  # Removido
    ]

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        corpora: Corpora | pd.DataFrame,
        corpora_metadata: dict[str, CorporaMetadata] | None = None,
        docs_col: str = "document",
        spacy_language_model: str = "pt_core_news_sm",
        spacy_pipe_components: None
        | list[
            Literal[
                "tok2vec",
                "morphologizer",
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "ner",
            ]
        ] = ["tok2vec", "morphologizer"],
    ) -> None:
        print("\n🚀 Initializing NoiseReducer object ...")
        self.corpora = self._preprocess_corpora(corpora)
        self.corpora_metadata = corpora_metadata
        self.docs_col = docs_col
        self.spacy_language_model = spacy_language_model
        self.no_io_indices = None
        self._build_corpora_nlp_docs(enable=spacy_pipe_components or [])
        self._build_metadata_feature()
        self._rules = self._build_rules()
        print("Object init done 🛬")

    def _preprocess_corpora(self, corpora: Corpora | pd.DataFrame) -> pd.DataFrame:
        if type(corpora) == Corpora:
            corpora = pd.DataFrame(asdict(corpora))
        corpora = CorporaDataFrame.validate(corpora)
        corpora["text"] = corpora.text.str.strip()
        return corpora

    def _build_corpora_nlp_docs(self, enable: list[str]) -> None:
        try:
            assert (
                type(self.corpora.nlp_doc.iloc[0]) == spacy.tokens.doc.Doc
            ), "Incorrect type for 'nlp_doc' feature."

            doc = self.corpora.nlp_doc.iloc[0]
            assert (
                doc[0].has_morph() and doc[0].has_vector
            ), "`nlp_doc` was created without the required spaCy's pipeline components. Recreating it..."

        except AttributeError as e:
            print(e)
            self._execute_doc_creation(enable, after_col="text")
        except AssertionError as e:
            print(e)
            self._execute_doc_creation(enable)

    def _build_metadata_feature(self) -> None:
        try:
            assert (
                type(self.corpora.metadata.iloc[0]) == dict
            ), "Incorrect type for 'metadata' feature."
            _ = self.corpora.metadata.apply(lambda x: x.update({"nr_tags": []}))

        except AttributeError as e:
            print(e)
            self.corpora["metadata"] = [{"nr_tags": []}] * self.corpora.shape[0]
        except AssertionError as e:
            print(e)
            print("Created new feature 'metadata_nr'.")
            self.corpora["metadata_nr"] = [{"nr_tags": []}] * self.corpora.shape[0]

    def _execute_doc_creation(
        self, enable: list[str], after_col: str | None = None
    ) -> None:
        print("Loading spacy language model and creating `spacy.Doc` type feature.")
        nlp = spacy.load(self.spacy_language_model)
        disabled_components = [pipe for pipe in nlp.pipe_names if pipe not in enable]
        nlp_doc = list(nlp.pipe(self.corpora.text, disable=disabled_components))

        if after_col:
            self.corpora.insert(
                loc=(self.corpora.columns.get_loc(after_col) + 1),
                column="nlp_doc",
                value=nlp_doc,
            )
        else:
            self.corpora["nlp_doc"] = nlp_doc

    @property
    def active_rules(self) -> list[str]:
        return [
            rule_name
            for rule_name, rule_config in self._rules.items()
            if rule_config.active
        ]

    def _build_rules(self) -> dict[str, _RuleConfig]:
        return {
            rule.value: _RuleConfig(active=True, fn=self._get_rule_fn(rule.value))
            for rule in Rules
        }

    def _get_rule_fn(self, rule_name: str) -> Callable:
        match rule_name:
            case Rules.TITLES.value:
                return self.find_titles
            case Rules.SUBTITLES.value:
                return self.find_subtitles
            case Rules.RANDOM_SUBTITLES.value:
                return self.find_random_subtitles
            case Rules.INTRO_PASSAGES.value:
                return self.find_intro_passages
            case Rules.REPUBLISHED_PASSAGES.value:
                return self.find_republished_passages
            case Rules.IRRELEVANT_ANNEXES_PASSAGES.value:
                return self.find_irrelevant_annexes
            case Rules.NO_IO_ARTICLES.value:
                return self.find_no_io_articles

            # TODO: Review the naming for `IRRELEVANT_MISC_PASSAGES`. Something
            # on the lines of "irrelevant individual passages", for instance.
            # case Rules.IRRELEVANT_MISC_PASSAGES.value:
            #     return self.find_irrelevant_misc_passages
            case Rules.NO_TEXT_PASSAGE.value:
                return self._search_no_text_passages
            case Rules.MISC_IRELEVANT_PASSAGE.value:
                return self._search_misc_irrelevant_passages
            case Rules.NOTES_PASSAGE.value:
                return self._search_notes_passages
            case Rules.ONE_WORD_PASSAGE.value:
                return self._search_one_word_passages
            case Rules.ALTERED_PASSAGE.value:
                return self._search_altered_passages

            case _:
                raise ValueError(f"'{rule_name}' does not have an associated rule.")

    def disable_rules(self, rules: list[str]) -> None:
        for rule_to_disable in rules:
            self._rules.update({rule_to_disable: _RuleConfig(active=False)})

    def reset_rules(self) -> None:
        self._rules = self._build_rules()

    # ===================================================================================== #
    # ROUTINE CALL
    # ===================================================================================== #
    def run_routine(
        self,
        handpicked_to_drop: list[Section] | None = None,
        handpicked_to_keep: list[Section] | None = None,
    ) -> list[int]:
        print("\n")
        print("Starting noise reduction routine ...")
        no_io_indices: list[int] = []
        doc_names = self.corpora[self.docs_col].unique()
        for doc_name in tqdm(doc_names):
            # print("=========================================================")
            # print(f"Processing doc_name '{doc_name}'")
            # print("=========================================================")

            df = self.corpora[self.corpora[self.docs_col] == doc_name].copy()
            indices = self.run_routine_on_document(
                df, doc_name, handpicked_to_drop, handpicked_to_keep
            )
            no_io_indices = [*no_io_indices, *indices]

        print("Routine completed 💪")
        self.no_io_indices = sorted(no_io_indices)
        return self.no_io_indices

    def run_routine_on_document(
        self,
        doc_df: pd.DataFrame,
        doc_name: str,
        handpicked_to_drop: list[Section] | None = None,
        handpicked_to_keep: list[Section] | None = None,
    ) -> list[int]:
        unique_doc = doc_df.document.unique()
        assert (
            unique_doc.shape[0] == 1
        ), "Multiple document sources found during processing."
        assert (
            unique_doc[0] == doc_name
        ), "`doc_name` does NOT match the `doc_df`'s document."

        # Get structural passages
        structural_passages = self.find_structural_passages(doc_df)
        passages_to_drop = [
            passages
            for structure, passages in structural_passages.items()
            if self._rules[structure].active
        ]

        # Store rules' metadata
        for structure, passages in structural_passages.items():
            self.update_metadata_tags(doc_df, passages.index, structure)

        for rule_name, rule_config in self._rules.items():
            if rule_config.active and rule_name not in NoiseReducer._STRUCTURAL_RULES:
                passages = rule_config.fn(
                    doc_df, structural_passages[Rules.TITLES.value]
                )
                passages_to_drop.append(passages)
                # passages_to_drop.append(
                #     rule_config.fn(doc_df, structural_passages[Rules.TITLES.value])
                # )
                self.update_metadata_tags(doc_df, passages.index, rule_name)

        # Handpicked sections to drop:
        if (
            handpicked_to_drop is not None
            and len(
                sections := [
                    section for section in handpicked_to_drop if section.doc == doc_name
                ]
            )
            > 0
        ):
            passages = [
                self.find_section(
                    doc_df, section, structural_passages[Rules.TITLES.value]
                )
                for section in sections
            ]
            passages_to_drop.extend(passages)
            self.update_metadata_tags(doc_df, passages, "handpicked")

        # Combine passages to drop
        drop_df = pd.concat(passages_to_drop)
        drop_df = drop_df[~drop_df.index.duplicated(keep="first")]

        # Handpicked sections to keep:
        if (
            handpicked_to_keep is not None
            and len(
                sections := [
                    section for section in handpicked_to_keep if section.doc == doc_name
                ]
            )
            > 0
        ):
            passages_to_keep = [
                self.find_section(
                    doc_df, section, structural_passages[Rules.TITLES.value]
                )
                for section in sections
            ]
            keep_df = pd.concat(passages_to_keep)
            keep_df = keep_df[~keep_df.index.duplicated(keep="first")]
            drop_df = drop_df[~drop_df.index.isin(keep_df.index)]

        return drop_df.index.tolist()

    def update_metadata_tags(self, df, indices, tag) -> None:
        df[df.index.isin(indices)].metadata.apply(lambda x: x["nr_tags"].append(tag))

    # ------------------------------------------------------------------------------------- #
    def remove_no_io_passages(
        self, no_io_indices: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Remove the identified indices without IOs (`no_io_indices`).
        """
        no_io_indices = no_io_indices or self.no_io_indices
        return self.corpora.drop(index=no_io_indices).reset_index(drop=True)

    # ------------------------------------------------------------------------------------- #
    def tidy_up(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Tidy up dataframe — i.e. remove the created `nlp_doc` feature.
        """
        df_ = df if (df is not None) else self.corpora
        return df_.drop(columns="nlp_doc")

    # ===================================================================================== #
    # STRUCTURAL RULES
    # ===================================================================================== #

    def find_structural_passages(self, df) -> dict[str, pd.Series]:
        titles = self._rules[Rules.TITLES.value].fn(df)
        subtitles = self._rules[Rules.SUBTITLES.value].fn(df, titles)
        random_subtitles = self._rules[Rules.RANDOM_SUBTITLES.value].fn(
            df, titles, subtitles
        )
        return {
            Rules.TITLES.value: titles,
            Rules.SUBTITLES.value: subtitles,
            Rules.RANDOM_SUBTITLES.value: random_subtitles,
        }

    # TODO: Refactor to return only indices (instead of text series)
    def find_titles(self, df) -> pd.Series:
        """Titles (e.g. "Capítulo", "Artigo", "Secção" e "Subsecção")."""
        pattern = "|".join([title_pattern.pat for title_pattern in CasedTitlesPatterns])
        return df.text[df.text.str.match(pat=pattern, case=True)]

    def _find_titles_subgroup(self, df, pattern_config: PatternConfig) -> pd.Series:
        """Find a specific subgroup of titles"""
        return df.text[
            df.text.str.match(pat=pattern_config.pat, case=pattern_config.case)
        ]

    # TODO: Refactor to return only indices (instead of text series)
    def find_subtitles(self, df, titles):
        """Subtitles"""
        # Start by selecting all passages following the identified titles.
        subtitles: pd.DataFrame = df[df.index.isin(titles.index + 1)].copy()

        # Skip numbered instances, because these are not actually subtitles (so
        # should NOT be removed).
        numbered_text = subtitles[subtitles.text.str.match(r"^\d", case=False)]
        subtitles = subtitles[~subtitles.index.isin(numbered_text.index)]

        # Skip instances ending with colon (":") – they are, virtually, not subtitles.
        colon_text = subtitles[subtitles.text.str.strip().str.endswith(":")]
        subtitles = subtitles[~subtitles.index.isin(colon_text.index)]

        # Skip instances that end with a dot ("." or ".»"), are double checked
        # as being a PUNCT (by spaCy), and start with specific POS tags.
        dot_subtitles = subtitles[
            subtitles.text.str.endswith((".", ".»"))
            & (subtitles.nlp_doc.apply(lambda doc: doc[-1].is_punct))
        ]
        ## POS tags identification:
        pos_to_check = ["DET", "ADP", "VERB", "AUX", "ADV"]
        dot_and_pos_text = dot_subtitles[
            dot_subtitles.nlp_doc.apply(lambda doc: doc[0].pos_ in pos_to_check)
        ]
        subtitles = subtitles[~subtitles.index.isin(dot_and_pos_text.index)]

        return subtitles.text

    def find_random_subtitles(self, df, titles, subtitles):
        """Random Subtitles"""
        random_subtitles = df.text[
            (~df.nlp_doc.apply(lambda x: self._safe_check_if_is_punct(x)))
            & (df.nlp_doc.apply(lambda x: self._safe_check_if_is_word(x)))
            & (
                ~df.text.str.match(pat=r"^([0-9]+[\)|\s\-]\s?|[a-z]+\)\s?)", case=True)
            )  # Check if starts with list ordering
        ]
        return random_subtitles[
            [
                index
                for index in random_subtitles.index
                if index not in titles.index.union(subtitles.index)
            ]
        ]

    # ===================================================================================== #
    # INDEPENDENT RULES
    # ===================================================================================== #

    # Intro passages
    def find_intro_passages(self, df, titles):
        """Intro passages"""
        return df.text[list(range(df.index[0], titles.index[0]))]

    # ===================================================================================== #

    # Republications
    def find_republished_passages(self, df, *args):
        republished_subtitle = self._search_republishing_subtitle(df.text, df.nlp_doc)
        if republished_subtitle.any():
            section_to_keep = "republished"
            # if self.corpora_metadata:
            if (
                self.corpora_metadata
                and (document := df[self.docs_col].iloc[0])
                in self.corpora_metadata.keys()
            ):
                # document = df[self.docs_col].iloc[0]
                section_to_keep = self.corpora_metadata[document].section_to_keep

            if section_to_keep == "initial":
                return df.text[
                    list(range(republished_subtitle.index[0], (df.index[-1] + 1)))
                ]
            elif section_to_keep == "republished":
                return df.text[list(range(df.index[0], republished_subtitle.index[0]))]
        return pd.Series(dtype="object")

    def _search_republishing_subtitle(self, text_series, nlp_docs_series):
        return text_series[
            (~nlp_docs_series.apply(lambda x: self._safe_check_if_is_punct(x)))
            & (nlp_docs_series.apply(lambda x: self._safe_check_if_is_word(x)))
            & (~text_series.str.match(pat=r"^([0-9]+\s\-\s?|[a-z]+\)\s?)", case=True))
            & (
                text_series.str.match(
                    pat="|".join(
                        [
                            r"^Republicação (.*) Decreto-Lei n\.(.*)$",
                        ]
                    )
                )
            )
            & (
                ~text_series.str.match(pat=r"^Regulamento interno$")
            )  # TODO: TEMPORARY SOLUTION; IMPROVE
        ]

    # ===================================================================================== #

    # Annexes that do NOT have articles within its content.
    def find_irrelevant_annexes(self, doc_df: pd.DataFrame, *args) -> pd.DataFrame:
        """Find indices of Annexes, which do NOT contain 'Articles'.

        Args:
            doc_df (pd.DataFrame): A specific document's corpus. Must contain feature "text".

        Returns:
            list[pd.Series]: Annexes' sections in which 'Article' titles were NOT found.
        """
        annex_titles = self._find_titles_subgroup(doc_df, CasedTitlesPatterns.ANNEXES)
        irrelevant_annexes = []
        for i, index in enumerate(annex_titles.index):
            j = i + 1
            try:
                next_index = annex_titles.index[j]
            except IndexError:
                next_index = doc_df.index[-1] + 1
            finally:
                doc_annex_df = doc_df.loc[index:next_index]
                articles = self._find_titles_subgroup(
                    doc_annex_df, CasedTitlesPatterns.ARTICLES
                )
                bases = self._find_titles_subgroup(
                    doc_annex_df, CasedTitlesPatterns.BASES
                )

            if articles.empty and bases.empty:
                irrelevant_annexes.append(doc_annex_df.text)

        if len(irrelevant_annexes) == 0:
            return pd.Series()

        annexes_df = pd.concat(irrelevant_annexes)
        annexes_df = annexes_df[~annexes_df.index.duplicated(keep="first")]
        return annexes_df

    # ===================================================================================== #

    # Irrelevant passages
    # def find_irrelevant_misc_passages(self, df, *args):
    #     """
    #     Combined irrelevant passages
    #     """
    #     no_text_passages = self._search_no_text_passages(df)
    #     misc_irrelevant_passages = self._search_misc_irrelevant_passages(df)
    #     altered_passages = self._search_altered_passages(df)
    #     one_word_passages = self._search_one_word_passages(df)
    #     indices = self._combine_indices(
    #         no_text_passages,
    #         misc_irrelevant_passages,
    #         altered_passages,
    #         one_word_passages,
    #     )
    #     return df.text[indices]

    # --------------------------------------------------------------------------- #
    def _search_no_text_passages(self, df, *args):
        """
        Passages without text
        In other words, no letters — only numbers and special characters.
        """
        no_text_pattern = "|".join([".*[a-zA-Z].*"])
        return df.text[~df.text.str.match(pat=no_text_pattern)]

    # --------------------------------------------------------------------------- #
    def _search_misc_irrelevant_passages(self, df, *args):
        """
        Misc irrelevant passages
        Passages that serve as image caption, alt-labels for tables, a few random
        subtitles, and more...
        """
        pat = r"^\(.*\)$"
        return df.text[df.text.str.match(pat=pat)]

    # --------------------------------------------------------------------------- #
    def _search_notes_passages(self, df: pd.DataFrame | None = None, *args):
        if df is None:
            df = self.corpora
        pat = r"^Nota(\.|\s\d+\.?)?\s\-"
        return df[df.text.str.match(pat)].text

    # --------------------------------------------------------------------------- #
    def _search_altered_passages(self, df, *args):
        """
        'Unaltered' and 'Revoked' passages
        """
        pat = r"^([0-9]+\s\-\s?|[a-z]+\)\s?)?(\[.*\]|\(.*\))[:;»\.]?$"
        return df.text[df.text.str.match(pat=pat)]

    # --------------------------------------------------------------------------- #
    def _search_one_word_passages(self, df, *args):
        """
        'One-token' passages
        """

        def _is_one_word_fragment(nlp_doc):
            token_count = 0
            for token in nlp_doc:
                if not token.is_punct:
                    token_count += 1
            return token_count <= 1

        return df.text[df.nlp_doc.apply(_is_one_word_fragment)]

    # ===================================================================================== #

    # Entire articles without IOs
    def find_no_io_articles(self, df, titles):
        """
        Entire articles without IOs
        """
        found_subtitles_indices = self._find_subtitles_indices(
            df.text, NoiseReducer._NO_IO_ARTICLES
        )
        articles_range_indices_to_remove = self._find_articles_passages(
            df.text, found_subtitles_indices, titles.index.to_list()
        )
        articles_range_indices_to_remove = self._flatten_list_of_lists(
            articles_range_indices_to_remove
        )
        return df.text[articles_range_indices_to_remove]

    def _find_subtitles_indices(
        self, text_series: pd.Series, subtitles_to_search: list[str]
    ) -> list[int]:
        """
        Find indices that match the listed articles' subtitles
        """
        articles_to_remove_indices = []
        regex_articles_to_remove = [f"^{regex}$" for regex in subtitles_to_search]
        for regex in regex_articles_to_remove:
            matches = text_series.str.match(regex, case=False)
            if matches.any():
                articles_to_remove_indices.append(text_series[matches].index)
        return [
            index
            for pandas_indices in articles_to_remove_indices
            for index in pandas_indices.to_list()
        ]

    def _find_articles_passages(self, text_series, subtitles_indices, titles_indices):
        """
        Using the title index to identify the passages of articles to be removed
        """
        merged_indices = sorted(subtitles_indices + titles_indices)
        if merged_indices[-1] in subtitles_indices:
            merged_indices.append(text_series.index[-1] + 1)
        articles_range_indices_to_remove = []
        for subtitle_index in sorted(subtitles_indices):
            merged_subtitle_index = merged_indices.index(subtitle_index)
            try:
                stopping_index = merged_indices[merged_subtitle_index + 1]
            except Exception as e:
                print("Exception within `_find_articles_passages`")
                print(e)
                stopping_index = text_series.shape[0]  # - 1
            articles_range_indices_to_remove.append(
                list(range(subtitle_index, stopping_index))
            )
        return articles_range_indices_to_remove

    # ===================================================================================== #

    # Find 'hand-picked' sections
    def find_section(
        self,
        doc_df: pd.DataFrame,
        section: Section,
        doc_titles: pd.Series | None = None,
    ):
        if not section.delimiter:
            assert (
                doc_titles is not None
            ), "To identify a specific section (i.e., `Section.delimiter=False`), than `doc_titles` should be passed."
        section_matches = doc_df[doc_df.text.str.fullmatch(pat=section.pat)]
        starting_index = section_matches.iloc[section.idx].name
        ending_index = (
            (doc_df.index[-1] + 1)
            if section.delimiter
            else doc_titles.index[doc_titles.index > starting_index][0]
        )
        
        # To remove section's title -> ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        return doc_df.text[list(range((starting_index + 1), ending_index))]


    # ===================================================================================== #
    # UTILITY SCRIPTS
    # ===================================================================================== #

    def _safe_check_if_is_punct(self, nlp_doc: SpacyDoc) -> bool:
        # return nlp_doc[-1].is_punct
        try:
            is_punct = nlp_doc[-1].is_punct
        except IndexError:
            is_punct = True
        return is_punct

    # --------------------------------------------------------------------------- #
    def _safe_check_if_is_word(self, nlp_doc: SpacyDoc) -> bool:
        # return self._is_it_word_shape(nlp_doc[-1].shape_)
        try:
            is_word = self._is_it_word_shape(nlp_doc[-1].shape_)
        except IndexError:
            is_word = False
        return is_word

    def _is_it_word_shape(self, string_shape: str) -> bool:
        search = re.compile(r"[^Xx\-]").search
        return not bool(search(string_shape))

    # --------------------------------------------------------------------------- #
    # Utility function to combine indices (i.e. `pandas.Index`)
    def _combine_indices(self, *args) -> pd.Index:
        indices = pd.Index([])
        for arg in args:
            indices = indices.union(arg.index)
        return indices

    # --------------------------------------------------------------------------- #
    # Utility function to flatten nested lists
    def _flatten_list_of_lists(self, list_of_lists):
        return [item for item_list in list_of_lists for item in item_list]

    # --------------------------------------------------------------------------- #
    def _count_annotations(self, df):
        labels_list = [col for col in df.columns if col.split("_")[0] == "oi"]
        labels_sum = df[
            [col for col in labels_list if col.split("_")[-1] != "obs"]
        ].sum(axis=1)
        labels_sum[labels_sum > 1] = 1
        return labels_sum.sum()
