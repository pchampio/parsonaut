from parsonaut import Parsable


class ConfigWithComments(Parsable):
    def __init__(
        self,
        name: str = "test",         # user name
        age: int = 25,              # age in years
        value: float = 3.14,        # some value
    ):
        self.name = name
        self.age = age
        self.value = value


def test_parsable_extracts_comments_automatically():
    """Parsable should automatically extract inline comments as help text."""
    assert hasattr(ConfigWithComments, '__field_help__')
    field_help = ConfigWithComments.__field_help__
    
    assert field_help.get('name') == 'user name'
    assert field_help.get('age') == 'age in years'
    assert field_help.get('value') == 'some value'


def test_parsable_works_without_comments():
    """Parsable should work fine even without comments."""
    class ConfigNoComments(Parsable):
        def __init__(self, x: int = 1, y: int = 2):
            self.x = x
            self.y = y
    
    assert hasattr(ConfigNoComments, '__field_help__')
    # Should be empty dict
    assert ConfigNoComments.__field_help__ == {}
