import 'axios';

declare module 'axios' {
  export interface AxiosInstance {
    // This tells the IDE: "When I call post<T>, I get T back, not AxiosResponse<T>"
    post<T = any, R = T, D = any>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<R>;
    get<T = any, R = T, D = any>(url: string, config?: AxiosRequestConfig<D>): Promise<R>;
  }
}
