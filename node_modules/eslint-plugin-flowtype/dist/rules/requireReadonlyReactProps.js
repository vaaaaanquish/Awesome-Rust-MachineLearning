"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
const schema = [];
const reComponentName = /^(Pure)?Component$/;
const reReadOnly = /^\$(ReadOnly|FlowFixMe)$/;

const isReactComponent = node => {
  if (!node.superClass) {
    return false;
  }

  return (// class Foo extends Component { }
    // class Foo extends PureComponent { }
    node.superClass.type === 'Identifier' && reComponentName.test(node.superClass.name) || // class Foo extends React.Component { }
    // class Foo extends React.PureComponent { }
    node.superClass.type === 'MemberExpression' && node.superClass.object.name === 'React' && reComponentName.test(node.superClass.property.name)
  );
};

const isReadOnlyObjectType = node => {
  if (!node || node.type !== 'ObjectTypeAnnotation') {
    return false;
  } // we consider `{||}` to be ReadOnly since it's exact AND has no props


  if (node.exact && node.properties.length === 0) {
    return true;
  } // { +foo: ..., +bar: ..., ... }


  return node.properties.length > 0 && node.properties.every(prop => {
    return prop.variance && prop.variance.kind === 'plus';
  });
};

const isReadOnlyType = node => {
  return node.right.id && reReadOnly.test(node.right.id.name) || isReadOnlyObjectType(node.right);
};

const create = context => {
  const readOnlyTypes = [];
  const foundTypes = [];
  const reportedFunctionalComponents = [];

  const isReadOnlyClassProp = node => {
    const id = node.superTypeParameters && node.superTypeParameters.params[0].id;
    return id && !reReadOnly.test(id.name) && !readOnlyTypes.includes(id.name) && foundTypes.includes(id.name);
  };

  for (const node of context.getSourceCode().ast.body) {
    let idName;
    let typeNode; // type Props = $ReadOnly<{}>

    if (node.type === 'TypeAlias') {
      idName = node.id.name;
      typeNode = node; // export type Props = $ReadOnly<{}>
    } else if (node.type === 'ExportNamedDeclaration' && node.declaration && node.declaration.type === 'TypeAlias') {
      idName = node.declaration.id.name;
      typeNode = node.declaration;
    }

    if (idName) {
      foundTypes.push(idName);

      if (isReadOnlyType(typeNode)) {
        readOnlyTypes.push(idName);
      }
    }
  }

  return {
    // class components
    ClassDeclaration(node) {
      if (isReactComponent(node) && isReadOnlyClassProp(node)) {
        context.report({
          message: node.superTypeParameters.params[0].id.name + ' must be $ReadOnly',
          node
        });
      } else if (node.superTypeParameters && node.superTypeParameters.params[0].type === 'ObjectTypeAnnotation' && !isReadOnlyObjectType(node.superTypeParameters.params[0])) {
        context.report({
          message: node.id.name + ' class props must be $ReadOnly',
          node
        });
      }
    },

    // functional components
    JSXElement(node) {
      let currentNode = node;
      let identifier;
      let typeAnnotation;

      while (currentNode && currentNode.type !== 'FunctionDeclaration') {
        currentNode = currentNode.parent;
      } // functional components can only have 1 param


      if (!currentNode || currentNode.params.length !== 1) {
        return;
      }

      if (currentNode.params[0].type === 'Identifier' && (typeAnnotation = currentNode.params[0].typeAnnotation)) {
        if ((identifier = typeAnnotation.typeAnnotation.id) && foundTypes.includes(identifier.name) && !readOnlyTypes.includes(identifier.name) && !reReadOnly.test(identifier.name)) {
          if (reportedFunctionalComponents.includes(identifier)) {
            return;
          }

          context.report({
            message: identifier.name + ' must be $ReadOnly',
            node
          });
          reportedFunctionalComponents.push(identifier);
          return;
        }

        if (typeAnnotation.typeAnnotation.type === 'ObjectTypeAnnotation' && !isReadOnlyObjectType(typeAnnotation.typeAnnotation)) {
          context.report({
            message: currentNode.id.name + ' component props must be $ReadOnly',
            node
          });
        }
      }
    }

  };
};

var _default = {
  create,
  schema
};
exports.default = _default;
module.exports = exports.default;