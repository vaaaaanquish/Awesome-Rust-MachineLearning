'use strict';

var GetIntrinsic = require('get-intrinsic');

var $TypeError = GetIntrinsic('%TypeError%');

var Get = require('./Get');
var ToLength = require('./ToLength');
var Type = require('./Type');

// https://ecma-international.org/ecma-262/11.0/#sec-lengthofarraylike

module.exports = function LengthOfArrayLike(obj) {
	if (Type(obj) !== 'Object') {
		throw new $TypeError('Assertion failed: `obj` must be an Object');
	}
	return ToLength(Get(obj, 'length'));
};

// TODO: use this all over
