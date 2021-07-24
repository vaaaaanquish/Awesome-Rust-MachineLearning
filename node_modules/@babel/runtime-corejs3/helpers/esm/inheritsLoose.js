import _Object$create from "@babel/runtime-corejs3/core-js/object/create";
import setPrototypeOf from "@babel/runtime-corejs3/helpers/esm/setPrototypeOf";
export default function _inheritsLoose(subClass, superClass) {
  subClass.prototype = _Object$create(superClass.prototype);
  subClass.prototype.constructor = subClass;
  setPrototypeOf(subClass, superClass);
}