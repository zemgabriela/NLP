from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures


class ExtendedFeatures(IDFeatures):
    """
    Class to add various emission features for a given word in a sequence.

    Function:
    add_emission_features: Adds various emission features based on the given word in the sequence.

    Input:
    - sequence: The sequence of words.
    - pos: The position of the current word in the sequence.
    - y: The tag ID for the current word.
    - features: The list to which features will be appended.

    Output:
    - The updated list of features with new features appended.
    """

    def add_emission_features(self, sequence, pos, y, features):
        # Get the word and tag name from the sequence
        word = str(sequence.x[pos])
        tag_name = self.dataset.y_dict.get_label_name(y)

        # List to hold features to add
        features_to_add = [
            ("id:%s::%s" % (word, tag_name), word),
            ("capi_ini::%s" % tag_name, word[0].isupper()),
            ("digit::%s" % tag_name, word.isdigit()),
            ("insidedigit::%s" % tag_name, any(char.isdigit() for char in word) and not word.isdigit()),
        ]

        # Dictionary of characters to features
        char_to_features = {
            '.': 'inside_point',
            '-': 'hyphen'
        }

        # Add features based on characters
        for char, feature_prefix in char_to_features.items():
            features_to_add.append((f"{feature_prefix}::{tag_name}", char in word))

        # List of suffixes
        suffixes = ['ing', 'ed', 'ness', 'ship', 'ity', 'ty', 'ly']

        # Add features based on suffixes
        for suffix in suffixes:
            features_to_add.append((f"ending_{suffix}::{tag_name}", word.endswith(suffix)))

        # Dictionary of words to features
        words_to_features = {
            'to': 'prep_to',
            'of': 'prep_of',
            'from': 'prep_from',
            'the': 'article_the'
        }

        # Add features based on specific words
        if word in words_to_features:
            features_to_add.append((f"{words_to_features[word]}::{tag_name}", True))

        # Process features and add them to the list if conditions are met
        for feature_name, condition in features_to_add:
            if condition:
                feature_id = self.add_feature(feature_name)
                if feature_id != -1:
                    features.append(feature_id)

        # Add feature for words that are not entirely lowercase after the first letter
        if len(word) > 1 and not word[1:].islower():
            feature_name = f"capi_ini::%s" % tag_name
            feature_id = self.add_feature(feature_name)
            if feature_id != -1:
                features.append(feature_id)

        return features



class ExtendedUnicodeFeatures(UnicodeFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        # Get the word and tag name from the sequence
        word = str(sequence.x[pos])
        tag_name = y

        # List to hold features
        features = []

        # Add a feature based on the word ID
        feature_name = f"id:{word}::{tag_name}"
        feature_id = self.add_feature(feature_name)
        if feature_id != -1:
            features.append(feature_id)

        # Check if the word is title-cased (first letter uppercase)
        if word.istitle():
            feature_name = f"uppercased::{tag_name}"
            feature_id = self.add_feature(feature_name)
            if feature_id != -1:
                features.append(feature_id)

        # Check if the word is a digit
        if word.isdigit():
            feature_name = f"number::{tag_name}"
            feature_id = self.add_feature(feature_name)
            if feature_id != -1:
                features.append(feature_id)

        # Check if the word contains a hyphen
        if '-' in word:
            feature_name = f"hyphen::{tag_name}"
            feature_id = self.add_feature(feature_name)
            if feature_id != -1:
                features.append(feature_id)

        # Add features for suffixes (up to 3 characters)
        for i in range(1, 4):
            if len(word) > i:
                suffix = word[-i:]
                feature_name = f"suffix:{suffix}::{tag_name}"
                feature_id = self.add_feature(feature_name)
                if feature_id != -1:
                    features.append(feature_id)

        # Add features for prefixes (up to 3 characters)
        for i in range(1, 4):
            if len(word) > i:
                prefix = word[:i]
                feature_name = f"prefix:{prefix}::{tag_name}"
                feature_id = self.add_feature(feature_name)
                if feature_id != -1:
                    features.append(feature_id)

        return features