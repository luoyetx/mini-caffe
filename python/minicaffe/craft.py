# coding = utf-8
# pylint: disable=too-many-arguments, invalid-name
"""Crafter for generate prototxt string"""


class LayerCrafter(object):
    """Layer Crafter for layer prototxt generation
    """

    def __init__(self, **kwargs):
        """parameters for this layer

        Parameters
        ----------
        name: string, required
            name of this layer
        type: string, required
            type of this layer, Input, Convolution, ...
        bottom: list(string), optional
            list of input blob name
        top: list(string), optional
            list of output blob name
        params: dict, optional
            extra parameters
        """
        assert 'name' in kwargs
        assert 'type' in kwargs
        self.params = kwargs

    def gen(self):
        """generate prototxt for this layer

        Returns
        -------
        prototxt: string
            prototxt for this layer
        """
        prototxt = self.parse_key_value('layer', self.params)
        return prototxt

    def parse_key_value(self, key, value, indent=''):
        """parse a key value pair to prototxt string, value can be some type

        Parameters
        ----------
        key: string
            key
        value: string, int, float, bool, list, dict
            value to be parsed
            string, int, float, bool: directly parsed
            list: parse and yield every element
            dict: parse and yield every key value pair
        indent: string
            indent for the line

        Returns
        -------
        s: string
            parsed prototxt string
        """
        if isinstance(value, str):
            return '%s%s: "%s"\n'%(indent, key, value)
        elif isinstance(value, bool):
            return '%s%s: %s\n'%(indent, key, str(value).lower())
        elif isinstance(value, int):
            return '%s%s: %d\n'%(indent, key, value)
        elif isinstance(value, float):
            return '%s%s: %f\n'%(indent, key, value)
        elif isinstance(value, list):
            s = ""
            for v in value:
                s += self.parse_key_value(key, v, indent)
            return s
        elif isinstance(value, dict):
            s = "%s%s {\n"%(indent, key)
            for key, val in list(value.items()):
                s += self.parse_key_value(key, val, indent+'\t')
            s += "%s}\n"%(indent)
            return s
        else:
            raise ValueError("unsupported value: %s"%value)
