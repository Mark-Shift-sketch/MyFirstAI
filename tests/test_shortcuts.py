import pytest
from app.jarvis import JarvisAssistant
from app.friday import FridayAssistant


def test_time_shortcut():
    j = JarvisAssistant()
    resp = j.respond('jarvis what time is it')
    assert resp and len(str(resp)) > 0


def test_date_shortcut():
    f = FridayAssistant()
    resp = f.respond("friday what's the date")
    assert resp and '202' in str(resp) or ',' in str(resp)


def test_sunrise_shortcut():
    j = JarvisAssistant()
    resp = j.respond('jarvis sunrise in London')
    assert resp and len(str(resp)) > 0
