import setPrototypeOf from "@babel/runtime/helpers/esm/setPrototypeOf";
export default function _inheritsLoose(subClass, superClass) {
  subClass.prototype = Object.create(superClass.prototype);
  subClass.prototype.constructor = subClass;
  setPrototypeOf(subClass, superClass);
}