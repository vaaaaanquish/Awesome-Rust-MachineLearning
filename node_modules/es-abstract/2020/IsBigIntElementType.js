'use strict';

// https://ecma-international.org/ecma-262/11.0/#sec-isbigintelementtype

module.exports = function IsBigIntElementType(type) {
	return type === 'BigUint64' || type === 'BigInt64';
};
