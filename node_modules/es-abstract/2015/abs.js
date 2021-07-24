'use strict';

var GetIntrinsic = require('get-intrinsic');

var $abs = GetIntrinsic('%Math.abs%');

// http://ecma-international.org/ecma-262/5.1/#sec-5.2

module.exports = function abs(x) {
	return $abs(x);
};
