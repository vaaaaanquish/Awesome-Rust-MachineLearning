import React, { Component } from 'react'
import ReactMarkdown from 'react-markdown'
import termsFrPath from './README.md'

class TopLayout extends Component {
  constructor(props) {
    super(props)

    this.state = { terms: null }
  }

  componentWillMount() {
    fetch(termsFrPath).then((response) => response.text()).then((text) => {
      this.setState({ terms: text })
    })
  }

  render() {
    return (
      <div className="content">
        <ReactMarkdown source={this.state.terms} />
      </div>
    )
  }
}

export default TopLayout
