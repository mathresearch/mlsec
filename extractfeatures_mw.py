import re
import numpy as np

chars = r" -~"
min_length = 5
string_regexp = '[%s]{%d,}' % (chars, min_length)


def _extract_features(self, string, hash_dim, split_regex= string_regexp):
        tokens = re.split(pattern=split_regex, string=string)
        hash_buckets = [(mmh3.hash(w) % hash_dim) for w in tokens]
        buckets, counts = np.unique(hash_buckets, return_counts=True)
        feature_values = np.zeros(hash_dim)
        for bucket, count in zip(buckets, counts):
            feature_values[bucket] = count
        return feature_values