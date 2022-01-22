import importlib
import logging
from typing import Optional

import spacy
from dbcat.catalog import CatColumn
from dbcat.catalog.pii_types import PiiType
from piicatcher import Address, Person, BirthDate
from piicatcher.detectors import register_detector, DatumDetector


LOGGER = logging.getLogger(__name__)


@register_detector
class SpacyDetector(DatumDetector):
    pii_cls_map = {
        'FAC': Address,      # Buildings, airports, highways, bridges, etc.
        'GPE': Address,      # Countries, cities, states.
        'LOC': Address,      # Non-GPE locations, mountain ranges, bodies of water.
        'PERSON': Person,       # People, including fictional.
        'PER': Person,          # Bug in french model
        'DATE': BirthDate,  # Dates within the period 18 to 100 years ago.
    }
    name = 'DatumSpacyDetector'

    def __init__(self, model: str = "en_core_web_md"):
        super(SpacyDetector, self).__init__()

        # Fixes a warning message from transformers that is pulled in via spacy
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.check_spacy_version()

        if not self.check_spacy_model(model):
            raise ValueError("Unable to find spacy model '{}'. Is your language supported? "
                             "Check the list of models available here: "
                             "https://github.com/explosion/spacy-models ".format(self.model))

        self.nlp = spacy.load(model)

        # If the model doesn't support named entity recognition
        if 'ner' not in [step[0] for step in self.nlp.pipeline]:
            raise ValueError(
                "The spacy model '{}' doesn't support named entity recognition, "
                "please choose another model.".format(self.model)
            )

    @staticmethod
    def check_spacy_version() -> bool:
        """Ensure that the version of spaCy is v3."""
        spacy_version = spacy.__version__  # spacy_info.get('spaCy version', spacy_info.get('spacy_version', None))

        if spacy_version is None:
            raise ImportError('Spacy v3 needs to be installed. Unable to detect spacy version.')
        try:
            spacy_major = int(spacy_version.split('.')[0])
        except Exception:
            raise ImportError('Spacy v3 needs to be installed. Spacy version {} is unknown.'.format(spacy_version))
        if spacy_major != 3:
            raise ImportError('Spacy v3 needs to be installed. Detected version {}.'.format(spacy_version))

        return True

    @staticmethod
    def check_spacy_model(model) -> bool:
        """Ensure that the spaCy model is installed."""
        spacy_info = spacy.info()
        if isinstance(spacy_info, str):
            raise ValueError('Unable to detect spacy models.')
        models = list(spacy_info.get('pipelines', spacy_info.get('models', None)).keys())
        if models is None:
            raise ValueError('Unable to detect spacy models.')

        if model not in models:
            LOGGER.info("Downloading spacy model {}".format(model))
            spacy.cli.download(model)
            importlib.import_module(model)
            # spacy.info() doesnt update after a spacy.cli.download, so theres no point checking it
            models.append(model)

        # Always returns true, if it fails to download, spacy sys.exit()s
        return model in models

    def detect(self, column: CatColumn, datum: str) -> Optional[PiiType]:
        doc = self.nlp(datum)
        for ent in doc.ents:
            LOGGER.debug("Found %s", ent.label_)
            if ent.label_ == "PERSON":
                return Person()

            if ent.label_ == "GPE":
                return Address()

            if ent.label_ == "DATE":
                return BirthDate()
