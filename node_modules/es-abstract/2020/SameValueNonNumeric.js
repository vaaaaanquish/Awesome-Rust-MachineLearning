'use strict';

var GetIntrinsic = require('get-intrinsic');

var $TypeError = GetIntrinsic('%TypeError%');

var SameValue = require('./SameValue');
var Type = require('./Type');

// https://ecma-international.org/ecma-262/11.0/#sec-samevaluenonnumeric

module.exports = function SameValueNonNumeric(x, y) {
	var xType = Type(x);
	if (xType === 'Number' || xType === 'Bigint') {
		throw new $TypeError('Assertion failed: SameValueNonNumeric does not accept Number or BigInt values');
	}
	if (xType !== Type(y)) {
		throw new $TypeError('SameValueNonNumeric requires two non-numeric values of the same type.');
	}
	return SameValue(x, y);
};
