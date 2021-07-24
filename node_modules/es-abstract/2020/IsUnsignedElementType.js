'use strict';

// https://ecma-international.org/ecma-262/11.0/#sec-isunsignedelementtype

module.exports = function IsUnsignedElementType(type) {
	return type === 'Uint8'
        || type === 'Uint8C'
        || type === 'Uint16'
        || type === 'Uint32'
        || type === 'BigUint64';
};
