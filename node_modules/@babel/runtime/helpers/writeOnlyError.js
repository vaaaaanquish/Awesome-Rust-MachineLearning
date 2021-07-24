function _writeOnlyError(name) {
  throw new TypeError("\"" + name + "\" is write-only");
}

module.exports = _writeOnlyError;