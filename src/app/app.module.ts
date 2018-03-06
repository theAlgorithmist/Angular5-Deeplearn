import { BrowserModule } from '@angular/platform-browser';
import { NgModule      } from '@angular/core';

import { AppComponent            } from './app.component';
import { CanvasSelectorDirective } from "./canvas-selector/canvas-selector.directive";

@NgModule({
  declarations: [
    AppComponent, CanvasSelectorDirective
  ],
  imports: [
    BrowserModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
