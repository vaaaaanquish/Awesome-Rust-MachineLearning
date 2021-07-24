// base: https://stackoverflow.com/a/51003410
import React, { Component } from 'react'
import ReactMarkdown from 'react-markdown'
import termsFrPath from './README.md'


function flatten(text, child) {
    return typeof child === 'string'
        ? text + child
        : React.Children.toArray(child.props.children).reduce(flatten, text)
}

function HeadingRenderer(props) {
    var children = React.Children.toArray(props.children)
    var text = children.reduce(flatten, '')
    var slug = text.toLowerCase().replace(/\W/g, '-')
    return React.createElement('h' + props.level, {id: slug}, props.children)
}

function changeURL(text) {
    var rt = text.replace(/(github.com\/vaaaaanquish)\/(.*#.*)/g, 'vaaaaanquish.github.io/$2')
    return rt
}

class TopLayout extends Component {
  constructor(props) {
    super(props)

    this.state = { terms: null }
  }

  componentWillMount() {
    fetch(termsFrPath).then((response) => response.text()).then((text) => {
      this.setState({ terms: changeURL(text) })
    })
  }

  render() {
    return (
      <div className="content">
        <ReactMarkdown children={this.state.terms} escapeHtml={false} components={{
            h1: HeadingRenderer,
            h2: HeadingRenderer,
            h3: HeadingRenderer,
            h4: HeadingRenderer
            }}/>
      </div>
    )
  }
}

export default TopLayout
