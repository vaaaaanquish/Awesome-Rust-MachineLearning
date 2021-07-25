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
    var slug = text.toLowerCase().replace(/\(|\)|&/g, '').replace(/\W/g, '-')
    return React.createElement('h' + props.level, {id: slug}, props.children)
}

function changeURL(text) {
    var rt = text.replace(/(https:\/\/github.com\/vaaaaanquish.*)#(.*)/g, '#$2')
    return rt
}

const styleButton = {
    margin: 0,
    top: 'auto',
    right: 40,
    bottom: 40,
    left: 'auto',
    position: 'fixed',
    borderRadius: '50%',
    height: '100px',
    width: '100px',
};

const styleLink = {
    display: 'block',
    height: '100%',
    paddingTop: '25px',
    fontSize: '40px',
};

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
        <ReactMarkdown children={this.state.terms} components={{
            h1: HeadingRenderer,
            h2: HeadingRenderer,
            h3: HeadingRenderer,
            h4: HeadingRenderer,
            img: ({node, ...props}) => <img id='awesome-rust-machinelearning' alt='arml' style={{width: '100%'}} {...props} />
            }}/>
      <button style={styleButton}><a href='#awesome-rust-machinelearning' style={styleLink}>Top</a></button>
      </div>
    )
  }
}

export default TopLayout
