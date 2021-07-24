export default ESLintWebpackPlugin;
export type Compiler = import('webpack').Compiler;
export type Options = import('./options').PluginOptions &
  import('eslint').ESLint.Options;
declare class ESLintWebpackPlugin {
  /**
   * @param {Options} options
   */
  constructor(options?: Options);
  options: import('./options').PluginOptions;
  /**
   * @param {Compiler} compiler
   */
  run(compiler: Compiler): Promise<void>;
  /**
   * @param {Compiler} compiler
   * @returns {void}
   */
  apply(compiler: Compiler): void;
  key: string | undefined;
  /**
   *
   * @param {Compiler} compiler
   * @returns {string}
   */
  getContext(compiler: Compiler): string;
}
