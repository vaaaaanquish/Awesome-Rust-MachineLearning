'use strict';

var IsInteger = require('./IsInteger');

// https://ecma-international.org/ecma-262/11.0/#sec-isnonnegativeinteger

module.exports = function IsNonNegativeInteger(argument) {
	return !!IsInteger(argument) && argument >= 0;
};
