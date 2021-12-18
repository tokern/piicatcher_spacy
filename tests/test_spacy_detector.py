from unittest.mock import patch

import pytest
from piicatcher import BirthDate, Person

from piicatcher_spacy.detectors.spacy import SpacyDetector


@pytest.mark.parametrize(
    'text, expected',
    [('Roger', Person),
     ('Jan 1 2016', BirthDate)]
)
def test_spacy(text, expected):
    detector = SpacyDetector(model='en_core_web_sm')
    with patch('piicatcher.scanner.CatColumn') as mocked:
        instance = mocked.return_value

        assert detector.detect(column=instance, datum=text) == expected()
