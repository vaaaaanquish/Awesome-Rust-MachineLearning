'use strict';

// https://ecma-international.org/ecma-262/11.0/#sec-isunclampedintegerelementtype

module.exports = function IsUnclampedIntegerElementType(type) {
	return type === 'Int8'
        || type === 'Uint8'
        || type === 'Int16'
        || type === 'Uint16'
        || type === 'Int32'
        || type === 'Uint32';
};
