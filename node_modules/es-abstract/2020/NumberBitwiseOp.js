'use strict';

var GetIntrinsic = require('get-intrinsic');

var $TypeError = GetIntrinsic('%TypeError%');

var ToInt32 = require('./ToInt32');
var ToUint32 = require('./ToUint32');

// https://ecma-international.org/ecma-262/11.0/#sec-numberbitwiseop

module.exports = function NumberBitwiseOp(op, x, y) {
	if (op !== '&' && op !== '|' && op !== '^') {
		throw new $TypeError('Assertion failed: `op` must be `&`, `|`, or `^`');
	}
	var lnum = ToInt32(x);
	var rnum = ToUint32(y);
	if (op === '&') {
		return lnum & rnum;
	}
	if (op === '|') {
		return lnum | rnum;
	}
	return lnum ^ rnum;
};
